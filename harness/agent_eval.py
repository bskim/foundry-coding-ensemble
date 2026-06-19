"""Agentic SWE-bench harness: run each config as an agent loop in the official
Docker image for the instance, and emit the resulting git diff as the prediction.

Unlike swe_eval.py (single-shot, oracle context), this runs the same instances
inside the SWE-bench Docker container for the instance (repo at base_commit, deps
installed, conda env `testbed`). Each turn the model emits one JSON tool call
(ls / read / grep / write / run pytest), which is executed via `docker exec`; the
observation is fed back. After the loop, `git diff` becomes the official
model_patch, scored by the same Docker harness (wsl_score.sh).

Predictions JSONL is the official format plus token fields summed over all turns
and a concatenated fusion_breakdown, so cost_report.py and wsl_score.sh consume
it unchanged. Runs on Windows and shells into WSL for Docker.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import uuid

from swe_eval import call_model, load_configs

HERE = os.path.dirname(os.path.abspath(__file__))

# Tolerate fences with or without a trailing newline (```json{...}``` on one line).
JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# SWE-bench instance image: pallets__flask-5014 -> ...pallets_1776_flask-5014
IMAGE_FMT = "swebench/sweb.eval.x86_64.{}:latest"


def image_for(instance_id):
    return IMAGE_FMT.format(instance_id.replace("__", "_1776_"))


# --------------------------------------------------------------------------- #
# WSL / Docker plumbing  (base64 everything so no quoting can break)
# --------------------------------------------------------------------------- #
def _wsl(cmd, timeout=120):
    """Run a bash -lc command string in WSL as root; return (rc, combined_output).

    Capture bytes and decode as UTF-8 (errors="replace"): WSL/container output is
    UTF-8, but subprocess `text=True` would decode with the Windows locale (cp949
    on Korean systems) and crash the reader thread on any non-cp949 byte (e.g. an
    em-dash or accented char in a source file), silently yielding an empty read.
    """
    p = subprocess.run(
        ["wsl.exe", "-u", "root", "-e", "bash", "-lc", cmd],
        capture_output=True, timeout=timeout,
    )
    out = (p.stdout or b"") + (p.stderr or b"")
    return p.returncode, out.decode("utf-8", "replace")


def _wsl_stdin(cmd, data, timeout=240):
    """Run a bash -lc command in WSL, feeding `data` (bytes) to its stdin.

    Large payloads (a whole file body for `write`) must travel through the process
    stdin, NOT the command line: a base64 blob on the wsl.exe argv overflows the
    Windows ~32KB command-line limit and fails with WinError 206.
    """
    p = subprocess.run(
        ["wsl.exe", "-u", "root", "-e", "bash", "-lc", cmd],
        input=data, capture_output=True, timeout=timeout,
    )
    out = (p.stdout or b"") + (p.stderr or b"")
    return p.returncode, out.decode("utf-8", "replace")


def ensure_docker():
    _wsl("service docker start >/dev/null 2>&1; for i in 1 2 3 4 5; do "
         "docker ps >/dev/null 2>&1 && break; sleep 2; done", timeout=60)


def exec_in(cid, script, timeout=180):
    """Pipe an arbitrary bash script into the container via stdin (no quote hell)."""
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    inner = f"echo {b64} | base64 -d | docker exec -i {cid} bash -l 2>&1"
    return _wsl(inner, timeout=timeout)


def start_container(instance_id):
    image = image_for(instance_id)
    # These images have CMD ["/bin/bash"] and no entrypoint; passing `sleep infinity`
    # as a positional CMD exits 255, so override the entrypoint to keep it alive.
    # The daemon is occasionally flaky right after start (container exits 255 / "not
    # running"), so verify it is actually Up and retry a couple of times.
    last = ""
    for attempt in range(3):
        ensure_docker()
        name = "agenteval_" + uuid.uuid4().hex[:10]
        rc, out = _wsl(f"docker run -d --entrypoint sleep --name {name} {image} infinity",
                       timeout=180)
        if rc == 0 and out.strip():
            st_rc, st = _wsl(f"docker inspect -f '{{{{.State.Running}}}}' {name} 2>&1",
                             timeout=30)
            if st_rc == 0 and st.strip() == "true":
                return name
            last = f"container not running after start: {st.strip()[:200]}"
        else:
            last = out.strip()[:300]
        _wsl(f"docker rm -f {name} >/dev/null 2>&1", timeout=60)
        time.sleep(3)
    raise RuntimeError(f"docker run failed for {image}: {last}")


def stop_container(name):
    _wsl(f"docker rm -f {name} >/dev/null 2>&1", timeout=60)


# --------------------------------------------------------------------------- #
# Tools  (executed inside the container, output capped to keep context small)
# --------------------------------------------------------------------------- #
READ_MAX_LINES = 220
GREP_MAX = 80
RUN_MAX_CHARS = 6000


def _b64(s):
    return base64.b64encode((s or "").encode("utf-8")).decode("ascii")


def tool_ls(cid, args):
    path = args.get("path", ".") or "."
    rc, out = exec_in(cid, f"cd /testbed && ls -la -- {_q(path)} 2>&1 | head -200")
    return out.strip()[:4000] or "(empty)"


def tool_read(cid, args):
    path = args.get("path", "")
    if not path:
        return "ERROR: read requires 'path'"
    start = int(args.get("start", 1) or 1)
    lines = int(args.get("lines", READ_MAX_LINES) or READ_MAX_LINES)
    lines = max(1, min(lines, READ_MAX_LINES))
    end = start + lines - 1
    rc, out = exec_in(
        cid,
        f"cd /testbed && nl -ba -- {_q(path)} 2>/dev/null | sed -n '{start},{end}p'")
    if not out.strip():
        return f"(no output; file may not exist or range empty): {path}"
    return out.rstrip()[:12000]


def tool_grep(cid, args):
    pattern = args.get("pattern", "")
    if not pattern:
        return "ERROR: grep requires 'pattern'"
    path = args.get("path", ".") or "."
    rc, out = exec_in(
        cid,
        f"cd /testbed && P=$(echo {_b64(pattern)} | base64 -d) && "
        f"grep -rnI -F \"$P\" -- {_q(path)} 2>/dev/null | head -{GREP_MAX}")
    return out.rstrip()[:8000] or "(no matches)"


def tool_write(cid, args):
    path = args.get("path", "")
    content = args.get("content")
    if not path or content is None:
        return "ERROR: write requires 'path' and 'content'"
    target = path if path.startswith("/") else "/testbed/" + path
    qt = _q(target)
    # Stream the file body through stdin (base64 on the Windows side, decoded in the
    # container) so a large file does not overflow the Windows command line (WinError
    # 206). Only the small path script travels on the command line.
    payload = base64.b64encode((content or "").encode("utf-8"))  # ascii bytes
    script = f"mkdir -p \"$(dirname {qt})\" && base64 -d > {qt}"
    inner = f"docker exec -i {cid} bash -c {_q(script)}"
    rc, out = _wsl_stdin(inner, payload, timeout=240)
    rc2, chk = exec_in(cid, f"wc -c < {qt} 2>/dev/null || echo MISSING")
    if rc != 0 or "MISSING" in chk:
        return "ERROR writing: " + (out.strip()[:400] or chk.strip()[:200] or "unknown")
    return f"ok: wrote {path} ({chk.strip()} bytes)"


def tool_run(cid, args):
    cmd = args.get("cmd", "")
    if not cmd:
        return "ERROR: run requires 'cmd'"
    rc, out = exec_in(cid, f"cd /testbed && C=$(echo {_b64(cmd)} | base64 -d) && "
                           f"timeout 180 bash -lc \"$C\" 2>&1", timeout=240)
    out = out.rstrip()
    if len(out) > RUN_MAX_CHARS:
        out = out[:RUN_MAX_CHARS // 2] + "\n...[truncated]...\n" + out[-RUN_MAX_CHARS // 2:]
    return f"(exit {rc})\n{out}" if out else f"(exit {rc}, no output)"


def _q(s):
    """Single-quote a path for bash."""
    return "'" + str(s).replace("'", "'\\''") + "'"


TOOLS = {
    "ls": tool_ls, "read": tool_read, "grep": tool_grep,
    "write": tool_write, "run": tool_run,
}


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
SYSTEM = (
    "You are an autonomous software engineer fixing a GitHub issue in a Python "
    "repository checked out at /testbed. You work by calling ONE tool per turn.\n\n"
    "Reply with EXACTLY one ```json fenced block and nothing else, of the form:\n"
    '```json\n{\"thought\": \"...\", \"tool\": \"<name>\", \"args\": {...}}\n```\n\n'
    "Tools:\n"
    "- ls   {\"path\": \"dir\"}                       list a directory\n"
    "- read {\"path\": \"f.py\", \"start\": 1, \"lines\": 200}  read file lines (numbered)\n"
    "- grep {\"pattern\": \"text\", \"path\": \".\"}      fixed-string search\n"
    "- write{\"path\": \"f.py\", \"content\": \"FULL new file contents\"}  overwrite a file\n"
    "- run  {\"cmd\": \"python -m pytest path::test -q\"}  run a shell command (180s cap)\n"
    "- finish {\"reason\": \"...\"}                    stop; your git diff is the patch\n\n"
    "Guidance:\n"
    "- Explore with grep/read first to locate the real cause; don't guess.\n"
    "- `python` and `pytest` already point to the project's env. cwd is /testbed.\n"
    "- Reproduce the bug, make the MINIMAL fix, then re-run the relevant tests.\n"
    "- `write` replaces the WHOLE file, so read it first and return it fully edited.\n"
    "- Call finish only when the fix is in place and tests you can run pass.\n"
    "- Keep changes to source; do not weaken or delete unrelated tests."
)


def _balanced_objects(text):
    """Return every top-level {...} substring, honoring nesting and strings.

    Reasoning models wrap the action in prose / chain-of-thought and sometimes emit
    several JSON-looking objects; a naive first-brace..last-brace slice then fails to
    parse. Scanning balanced braces lets us recover each candidate object instead.
    """
    objs = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, c in enumerate(text or ""):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objs.append(text[start:i + 1])
    return objs


def parse_action(text):
    text = text or ""
    candidates = [b.strip() for b in JSON_BLOCK.findall(text)]
    candidates.extend(_balanced_objects(text))
    # The model's final decision is the LAST valid action object; try from the end.
    for raw in reversed(candidates):
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("tool"):
            return obj, None
    return None, "no JSON object containing a \"tool\" field"


def accumulate_usage(acc, usage):
    acc["prompt_tokens"] += int((usage or {}).get("prompt_tokens", 0) or 0)
    acc["completion_tokens"] += int((usage or {}).get("completion_tokens", 0) or 0)
    acc["total_tokens"] += int((usage or {}).get("total_tokens", 0) or 0)
    bd = (usage or {}).get("_fusion_breakdown")
    if bd:
        acc["fusion_breakdown"].extend(bd)


def solve_instance(inst, cfg, max_turns, turn_timeout):
    instance_id = inst["instance_id"]
    name = start_container(instance_id)
    acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
           "fusion_breakdown": []}
    turns = 0
    finished = False
    dead_container = 0
    try:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"Repository: {instance_id.split('__')[0]}\n\n"
                f"## Issue\n{inst.get('problem_statement', '')}\n\n"
                "Begin. Locate the cause, fix it, and verify."},
        ]
        for turns in range(1, max_turns + 1):
            try:
                content, usage = call_model(cfg, messages, timeout=turn_timeout)
            except Exception as e:
                print(f"    turn {turns}: model error {e}")
                break
            accumulate_usage(acc, usage)
            action, err = parse_action(content)
            if err:
                messages.append({"role": "assistant", "content": content[:4000]})
                messages.append({"role": "user", "content":
                                 f"Could not parse a tool call ({err}). Reply with "
                                 "exactly one ```json block: {\"tool\":..., \"args\":...}."})
                continue
            tool = action.get("tool")
            targs = action.get("args") or {}
            if tool == "finish":
                finished = True
                print(f"    turn {turns}: finish ({targs.get('reason','')[:60]})")
                break
            fn = TOOLS.get(tool)
            if not fn:
                obs = f"ERROR: unknown tool '{tool}'. Valid: {', '.join(TOOLS)}, finish."
            else:
                try:
                    obs = fn(name, targs)
                except Exception as e:
                    obs = f"ERROR running {tool}: {e}"
            print(f"    turn {turns}: {tool} -> {str(obs)[:70].splitlines()[0] if obs else ''}")
            # If the container has died (flaky daemon), every tool call returns the
            # same daemon error; bail out instead of burning the whole turn budget.
            if isinstance(obs, str) and "Error response from daemon" in obs:
                dead_container += 1
                if dead_container >= 3:
                    print(f"    aborting: container unreachable for {dead_container} turns")
                    break
            else:
                dead_container = 0
            # keep only a compact transcript: assistant action + observation
            messages.append({"role": "assistant", "content": content[:4000]})
            obs_msg = f"Observation:\n{obs}"
            left = max_turns - turns
            if left <= 5:
                obs_msg += (f"\n\n[Only {left} turns left. If you have located the cause, "
                            "apply the fix now with `write`, run the test to confirm, then "
                            "`finish`. An empty git diff scores zero.]")
            messages.append({"role": "user", "content": obs_msg})

        # Extract the patch from the working tree.
        rc, diff = exec_in(name, "cd /testbed && git add -A && git diff --cached", timeout=120)
        patch = diff if (rc == 0 and diff.strip()) else ""
    finally:
        stop_container(name)

    return {
        "instance_id": instance_id,
        "model_name_or_path": cfg["name"],
        "model_patch": patch,
        "prompt_tokens": acc["prompt_tokens"],
        "completion_tokens": acc["completion_tokens"],
        "total_tokens": acc["total_tokens"],
        "turns": turns,
        "finished": finished,
        **({"fusion_breakdown": acc["fusion_breakdown"]} if acc["fusion_breakdown"] else {}),
    }


def main():
    ap = argparse.ArgumentParser(description="Agentic SWE-bench: fusion vs solo")
    ap.add_argument("--dataset", required=True, help="SWE-bench JSONL (instances)")
    ap.add_argument("--configs", default=os.path.join(HERE, "configs.fullset.json"))
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--turn-timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    configs = load_configs(args.configs)
    cfg = next((c for c in configs if c["name"] == args.config_name), None)
    if not cfg:
        ap.error(f"config '{args.config_name}' not found in {args.configs}")

    rows = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    ensure_docker()
    n_ok = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for inst in rows:
            t0 = time.time()
            print(f"=== {inst['instance_id']}  [{cfg['name']}] ===")
            try:
                rec = solve_instance(inst, cfg, args.max_turns, args.turn_timeout)
            except Exception as e:
                print(f"  FAILED: {e}")
                rec = {"instance_id": inst["instance_id"],
                       "model_name_or_path": cfg["name"], "model_patch": "",
                       "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                       "turns": 0, "finished": False, "error": str(e)}
            if rec.get("model_patch"):
                n_ok += 1
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"  -> patch={len(rec.get('model_patch',''))}c turns={rec.get('turns')} "
                  f"tok={rec.get('total_tokens')} {time.time()-t0:.0f}s")
    print(f"\nwrote {len(rows)} predictions ({n_ok} non-empty) -> {args.out}")


if __name__ == "__main__":
    main()
