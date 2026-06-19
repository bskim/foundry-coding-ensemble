"""Single-shot SWE-bench prediction generator (oracle-context).

Reads a SWE-bench JSONL dataset, asks a configured model to rewrite the
file(s) relevant to each issue, and writes predictions.jsonl in the official
format {instance_id, model_name_or_path, model_patch}. Token counts are added
as extra fields for cost accounting. Scoring is done separately by the Docker
harness (see wsl_score.sh).

Configs are OpenAI-compatible endpoints, so adding a model is one config line:
    {"name": "solo-x", "base": "http://host/v1", "model": "<slug>"}
"""
import argparse
import difflib
import json
import os
import re
import time
from urllib import request as urlreq
from urllib import error as urlerror

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("LITELLM_KEY", "sk-1234")

DEFAULT_CONFIGS = [
    {"name": "solo", "base": "http://127.0.0.1:4000/v1", "model": "kimi-k2",
     "temperature": 0},
]

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
DIFF_FENCE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
FILE_REWRITE = re.compile(
    r"#+\s*FILE:\s*(?P<path>\S+)\s*\n```(?:[a-zA-Z0-9_+-]*)?\s*\n(?P<body>.*?)```",
    re.DOTALL,
)


def call_model(cfg, messages, timeout=1800, retries=4):
    payload = {"model": cfg["model"], "messages": messages, "stream": False}
    if cfg.get("temperature") is not None:
        payload["temperature"] = cfg["temperature"]
    if cfg.get("max_tokens") is not None:
        payload["max_tokens"] = cfg["max_tokens"]
    url = cfg["base"].rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(retries + 1):
        req = urlreq.Request(url, data=body, method="POST")
        req.add_header("Authorization", "Bearer " + KEY)
        req.add_header("Content-Type", "application/json")
        for k, v in (cfg.get("headers") or {}).items():
            req.add_header(k, v)
        try:
            resp = urlreq.urlopen(req, timeout=timeout)
            data = json.loads(resp.read().decode("utf-8", "replace"))
            break
        except urlerror.HTTPError as e:
            # Back off and retry on rate limits (429) and transient gateway errors
            # instead of dropping the instance to an empty patch.
            if e.code in (429, 500, 502, 503) and attempt < retries:
                wait = min(60, 5 * (2 ** attempt))
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if retry_after and str(retry_after).isdigit():
                    wait = max(wait, int(retry_after))
                print(f"    [retry {attempt+1}/{retries}] HTTP {e.code}; sleeping {wait}s")
                time.sleep(wait)
                last_err = e
                continue
            raise
    else:
        raise last_err if last_err else RuntimeError("call_model: retries exhausted")
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or msg.get("reasoning_content") or ""
    usage = data.get("usage") or {}
    # Aggregator endpoints may return a per-stage token breakdown; carry it
    # through so the cost report can price each underlying model.
    fusion = data.get("fusion") or {}
    if fusion.get("breakdown"):
        usage = dict(usage)
        usage["_fusion_breakdown"] = fusion["breakdown"]
    return content, usage


def extract_file(text):
    """Return the largest code fence in the answer, or None."""
    blocks = [b.strip("\n") for b in CODE_FENCE.findall(text)]
    if not blocks:
        return None
    return max(blocks, key=len)


def extract_diff(text):
    """Return a unified diff from the answer, or an empty string."""
    blocks = [b for b in DIFF_FENCE.findall(text)]
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n"
    if "diff --git" in text or text.lstrip().startswith("--- "):
        return text.strip("\n") + "\n"
    return ""


def parse_file_rewrites(text, known_paths):
    """Extract {path: full_new_contents} from a model answer.

    Primary format is `### FILE: <path>` plus a fenced block. Falls back to a
    single code fence when exactly one file is in scope.
    """
    out = {}
    for m in FILE_REWRITE.finditer(text):
        path = m.group("path").strip().strip("`").lstrip("./")
        body = m.group("body")
        if body.endswith("\n"):
            body = body[:-1]
        out[path] = body
    if not out and len(known_paths) == 1:
        block = extract_file(text)
        if block is not None:
            out[known_paths[0]] = block
    return out


def make_file_diff(path, old, new):
    """Build a git-applyable unified diff for one file from full contents."""
    if old is None:
        old = ""
    if not old.endswith("\n") and old != "":
        old += "\n"
    if not new.endswith("\n") and new != "":
        new += "\n"
    if old == new:
        return ""
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    body = "".join(diff)
    if not body:
        return ""
    return f"diff --git a/{path} b/{path}\n" + body


def build_patch_from_rewrites(inst, text):
    """Turn a model's full-file rewrites into one diff using the original file
    contents in oracle_context, so the diff applies at base_commit."""
    ctx = inst.get("oracle_context") or {}
    rewrites = parse_file_rewrites(text, list(ctx.keys()))
    parts = []
    for path, new_content in rewrites.items():
        old = ctx.get(path)
        d = make_file_diff(path, old, new_content)
        if d:
            parts.append(d if d.endswith("\n") else d + "\n")
    return "".join(parts)


def build_diff_messages(inst):
    """Ask the model directly for a unified diff (fallback prompt)."""
    ctx = inst.get("oracle_context") or {}
    ctx_block = ""
    for path, content in ctx.items():
        ctx_block += f"\n### `{path}`\n```python\n{content}\n```\n"
    user = (
        "You are resolving a GitHub issue in a Python repository. Produce a patch "
        "that fixes it. Return ONLY a unified diff (git format) inside a single "
        "```diff code block, with correct file paths relative to the repo root.\n\n"
        f"## Issue\n{inst.get('problem_statement', '')}\n"
        f"{('## Relevant files' + ctx_block) if ctx_block else ''}"
    )
    return [{"role": "user", "content": user}]


def build_rewrite_messages(inst):
    """Ask for the complete corrected contents of each file that must change.

    The diff is then built with difflib against the original contents in
    oracle_context, so the patch always applies cleanly at base_commit. The
    same prompt is used for every config for a fair comparison.
    """
    ctx = inst.get("oracle_context") or {}
    ctx_block = ""
    for path, content in ctx.items():
        ctx_block += f"\n### `{path}`\n```python\n{content}\n```\n"
    user = (
        "You are resolving a GitHub issue in a Python repository. Below are the "
        "current contents of the file(s) most relevant to the issue.\n\n"
        "For EACH file you need to change, output its COMPLETE corrected contents "
        "in this EXACT format (repeat per file):\n"
        "### FILE: <path/relative/to/repo/root>\n"
        "```python\n<full corrected file contents>\n```\n\n"
        "Rules:\n"
        "- Output the ENTIRE file, not a snippet or diff.\n"
        "- Change only what is necessary to fix the issue; keep everything else "
        "byte-for-byte identical.\n"
        "- Use the exact path shown in the headers below.\n"
        "- Do not add commentary outside the FILE blocks.\n\n"
        f"## Issue\n{inst.get('problem_statement', '')}\n"
        f"{('## Current files' + ctx_block) if ctx_block else ''}"
    )
    return [{"role": "user", "content": user}]


def run_swebench(args, configs):
    cfg = next((c for c in configs if c["name"] == args.config_name), configs[-1])
    rows = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out_path = args.out or os.path.join(HERE, "predictions.jsonl")
    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for inst in rows:
            patch, usage, mode = "", {}, ""
            try:
                content, usage = call_model(cfg, build_rewrite_messages(inst))
                patch = build_patch_from_rewrites(inst, content)
                mode = "rewrite"
                if not patch:
                    patch = extract_diff(content)
                    mode = "diff-fallback" if patch else "none"
            except Exception as e:
                print(f"[ERR] {inst.get('instance_id')}: {e}")
            if patch:
                n_ok += 1
            rec = {
                "instance_id": inst.get("instance_id"),
                "model_name_or_path": cfg["name"],
                "model_patch": patch,
                "prompt_tokens": int((usage or {}).get("prompt_tokens", 0) or 0),
                "completion_tokens": int((usage or {}).get("completion_tokens", 0) or 0),
                "total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
            }
            if (usage or {}).get("_fusion_breakdown"):
                rec["fusion_breakdown"] = usage["_fusion_breakdown"]
            out.write(json.dumps(rec) + "\n")
            print(f"[{(mode or 'none').upper()}] {inst.get('instance_id')} "
                  f"({int((usage or {}).get('total_tokens', 0) or 0)} tok)")
    print(f"\nwrote {len(rows)} predictions ({n_ok} non-empty) -> {out_path}")


def load_configs(path):
    if not path:
        return DEFAULT_CONFIGS
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="SWE-bench single-shot prediction generator")
    ap.add_argument("--configs", help="JSON file: list of {name,base,model,...}")
    ap.add_argument("--config-name", default="solo",
                    help="which config generates predictions")
    ap.add_argument("--dataset", required=True, help="SWE-bench JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="cap number of instances")
    ap.add_argument("--out", help="output predictions jsonl path")
    args = ap.parse_args()

    configs = load_configs(args.configs)
    run_swebench(args, configs)


if __name__ == "__main__":
    main()
