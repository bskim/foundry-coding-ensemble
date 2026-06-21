"""Agentic SWE-bench harness: Fusion ensemble vs solo, in a real repo with tools.

Why this exists
---------------
`swe_eval.py --mode swebench` is a SINGLE-SHOT, oracle-context harness: the model
is handed the gold files and asked to rewrite them once, with no repo exploration
and no test feedback. That under-measures every model (frontier ones especially),
because today's coding products (Claude Code / Codex / Copilot) are agentic.

This harness runs the SAME instances and SAME configs (solo + fusion virtual
models) as an AGENT LOOP inside the official SWE-bench Docker image for the
instance (repo at base_commit, deps installed, conda env `testbed`):

    issue -> [ ls / read / grep / write / run(pytest) ]* -> git diff = prediction

Each turn the model emits ONE json tool call; we execute it via `docker exec`
into the instance container and feed back the observation. After the loop we take
`git diff` as the official `model_patch`, so the result is scored by the very same
Docker harness (wsl_score.sh) -> directly comparable resolved-rate + $/resolved.

For a fusion config every turn fans out panel->judge->synth, so the per-turn cost
is 5-7x a solo call and the agentic token economics become explicit.

Predictions JSONL is OFFICIAL format plus token fields (summed over all turns) and
a concatenated `fusion_breakdown`, exactly like swe_eval.py, so cost_report.py and
wsl_score.sh consume it unchanged.

Standard library only. Runs on Windows; shells into WSL for Docker.
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


# WSL / Docker plumbing: base64 everything so no quoting can break.
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


# Tools executed inside the container; output is capped to keep context small.
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
    # `write` creates NEW files only; targeted changes to existing files must go
    # through `edit`. A whole-file overwrite forces the model to regenerate the
    # entire file (correctness risk on big files) and bloats both the diff and the
    # turn's completion tokens, so we steer it to a snippet replace instead.
    _, exists = exec_in(cid, f"test -e {qt} && echo EXISTS || echo NEW")
    if "EXISTS" in exists:
        return (f"ERROR: '{path}' already exists. Use `edit` (exact-snippet replace) "
                "for changes to existing files; `write` only creates new files.")
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


# In-container edit body (ported from MiMo-Code's tool/edit.ts cascade). Runs after an
# exact-match miss: line-trimmed -> whitespace-normalized -> indentation-flexible ->
# block-anchor, applying the first fuzzy candidate that occurs UNIQUELY. This removes
# the brittle "snippet whitespace must be byte-perfect" penalty that otherwise burns
# turns for every model. A post-edit syntax check (builtin compile, writes nothing)
# returns a real error signal -- a poor-man's LSP diagnostic -- so the model gets
# genuine feedback instead of discovering breakage only at the next test run. Raw
# string: backslash sequences (\n, \s) must reach the container python literally.
_EDIT_BODY = r"""try:
    s=open(p,encoding='utf-8').read()
except Exception as e:
    print('ERROR open: '+str(e)); sys.exit()

def _line_trimmed(content, find):
    olines=content.split('\n'); slines=find.split('\n')
    if slines and slines[-1]=='': slines=slines[:-1]
    n=len(slines)
    if n==0: return
    for i in range(0, len(olines)-n+1):
        if all(olines[i+j].strip()==slines[j].strip() for j in range(n)):
            start=sum(len(olines[k])+1 for k in range(i)); end=start
            for k in range(n):
                end+=len(olines[i+k])
                if k<n-1: end+=1
            yield content[start:end]

def _ws_norm(content, find):
    import re as _re
    def norm(t): return _re.sub(r'\s+',' ',t).strip()
    nf=norm(find); lines=content.split('\n'); flines=find.split('\n')
    if len(flines)>1:
        for i in range(0, len(lines)-len(flines)+1):
            block='\n'.join(lines[i:i+len(flines)])
            if norm(block)==nf: yield block
    else:
        for line in lines:
            if norm(line)==nf: yield line

def _indent_flex(content, find):
    def si(t):
        ls=t.split('\n'); ne=[l for l in ls if l.strip()]
        if not ne: return t
        mi=min(len(l)-len(l.lstrip()) for l in ne)
        return '\n'.join((l if not l.strip() else l[mi:]) for l in ls)
    nf=si(find); clines=content.split('\n'); flines=find.split('\n')
    for i in range(0, len(clines)-len(flines)+1):
        block='\n'.join(clines[i:i+len(flines)])
        if si(block)==nf: yield block

def _block_anchor(content, find):
    olines=content.split('\n'); slines=find.split('\n')
    if slines and slines[-1]=='': slines=slines[:-1]
    if len(slines)<3: return
    first=slines[0].strip(); last=slines[-1].strip(); cands=[]
    for i in range(len(olines)):
        if olines[i].strip()!=first: continue
        for j in range(i+2, len(olines)):
            if olines[j].strip()==last: cands.append((i,j)); break
    if len(cands)!=1: return
    i,j=cands[0]; start=sum(len(olines[k])+1 for k in range(i)); end=start
    for k in range(i, j+1):
        end+=len(olines[k])
        if k<j: end+=1
    yield content[start:end]

def _uniq(content, cand):
    idx=content.find(cand)
    if idx==-1: return False
    return content.find(cand, idx+1)==-1

matched=None; strategy=None
c=s.count(old)
if c==1:
    matched=old; strategy='exact'
elif c>1:
    print('ERROR: old matches '+str(c)+' places; add surrounding context to make it unique.'); sys.exit()
else:
    for name, fn in (('line-trimmed',_line_trimmed),('whitespace',_ws_norm),('indentation',_indent_flex),('block-anchor',_block_anchor)):
        for cand in fn(s, old):
            if _uniq(s, cand): matched=cand; strategy=name; break
        if matched is not None: break
    if matched is None:
        print('ERROR: no exact or fuzzy match for old in '+p+'; copy a larger snippet incl. a few surrounding lines (exact whitespace not required, but anchors must be unique).'); sys.exit()

open(p,'w',encoding='utf-8').write(s.replace(matched,new,1))
msg='ok: edited '+p+('' if strategy=='exact' else ' [fuzzy match via '+strategy+']')
if p.endswith('.py'):
    try:
        compile(open(p,encoding='utf-8').read(), p, 'exec')
    except SyntaxError as e:
        msg+='\nWARNING: this edit left a Python syntax error (fix it): '+str(e)
print(msg)
"""


def tool_edit(cid, args):
    """Replace one snippet in an existing file (search/replace edit).

    This is the primitive real coding agents (Aider / Claude Code / Codex) use: it
    keeps changes surgical, the diff small, and the per-turn completion tokens low,
    instead of having the model regenerate a whole file. An exact unique `old` is
    tried first; on a miss a fuzzy cascade (ported from MiMo-Code) locates a unique
    near-match that differs only in whitespace/indentation, so a semantically correct
    edit is not wasted on a byte-perfect-snippet requirement. The replace runs inside
    the container in python; path/old/new travel as base64 so no quoting or non-ascii
    can break the shell. A post-edit syntax check returns real breakage feedback.
    """
    path = args.get("path", "")
    old = args.get("old")
    new = args.get("new")
    if not path or old is None or new is None:
        return "ERROR: edit requires 'path', 'old', and 'new'"
    target = path if path.startswith("/") else "/testbed/" + path
    py = (
        "import base64,sys\n"
        f"p=base64.b64decode('{_b64(target)}').decode('utf-8')\n"
        f"old=base64.b64decode('{_b64(old)}').decode('utf-8')\n"
        f"new=base64.b64decode('{_b64(new)}').decode('utf-8')\n"
        + _EDIT_BODY
    )
    script = "cd /testbed && python - <<'PYEOF'\n" + py + "PYEOF\n"
    rc, out = exec_in(cid, script)
    return out.strip()[:800] or "(no output)"


# In-container apply_patch body (ported faithfully from MiMo-Code's patch/index.ts:
# parsePatch + deriveNewContentsFromChunks + the 4-pass seekSequence matcher). This
# is the OpenAI/Codex patch envelope (*** Begin Patch / *** Update File / @@ / +-).
# gpt-5.x models are RL-trained to emit exactly this format, so offering it natively
# lets a SOTA model apply multi-hunk, multi-file edits in ONE turn instead of many
# brittle single-snippet edits. Offered to EVERY config (uniform tool set = fair).
# Two-phase: parse+derive ALL hunks first, then write, so a mid-patch failure leaves
# the tree untouched (atomic). Context/old-line matching degrades exact -> rstrip ->
# trim -> unicode-normalized, mirroring the reference. Raw string: backslash escapes
# (\n, \u2026, regex classes) must reach the container python literally.
_APPLY_PATCH_BODY = r"""
def _norm_uni(s):
    s=re.sub(r"[\u2018\u2019\u201A\u201B]","'",s)
    s=re.sub(r'[\u201C\u201D\u201E\u201F]','"',s)
    s=re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]","-",s)
    s=s.replace("\u2026","...").replace("\u00A0"," ")
    return s

def _eqm(a,b,mode):
    if mode==0: return a==b
    if mode==1: return a.rstrip()==b.rstrip()
    if mode==2: return a.strip()==b.strip()
    return _norm_uni(a.strip())==_norm_uni(b.strip())

def _try(lines,pat,start,mode,eof):
    n=len(pat)
    if eof:
        fe=len(lines)-n
        if fe>=start and all(_eqm(lines[fe+j],pat[j],mode) for j in range(n)): return fe
    for i in range(start,len(lines)-n+1):
        if all(_eqm(lines[i+j],pat[j],mode) for j in range(n)): return i
    return -1

def _seek(lines,pat,start,eof=False):
    if not pat: return -1
    for mode in (0,1,2,3):
        r=_try(lines,pat,start,mode,eof)
        if r!=-1: return r
    return -1

def _derive(orig,chunks):
    lines=orig.split("\n")
    if lines and lines[-1]=="": lines.pop()
    repls=[]; idx=0
    for ch in chunks:
        if ch["ctx"]:
            ci=_seek(lines,[ch["ctx"]],idx)
            if ci==-1: raise ValueError("context not found: "+ch["ctx"])
            idx=ci+1
        if len(ch["old"])==0:
            ins=len(lines)-1 if (lines and lines[-1]=="") else len(lines)
            repls.append((ins,0,ch["new"])); continue
        pat=ch["old"]; ns=ch["new"]
        f=_seek(lines,pat,idx,ch["eof"])
        if f==-1 and pat and pat[-1]=="":
            pat=pat[:-1]
            if ns and ns[-1]=="": ns=ns[:-1]
            f=_seek(lines,pat,idx,ch["eof"])
        if f==-1: raise ValueError("could not locate lines:\n"+"\n".join(ch["old"]))
        repls.append((f,len(pat),ns)); idx=f+len(pat)
    repls.sort(key=lambda r:r[0])
    res=list(lines)
    for s,ol,seg in reversed(repls):
        res[s:s+ol]=seg
    if not res or res[-1]!="": res.append("")
    return "\n".join(res)

def _parse(text):
    lines=text.strip().split("\n")
    b=next((k for k,l in enumerate(lines) if l.strip()=="*** Begin Patch"),-1)
    e=next((k for k,l in enumerate(lines) if l.strip()=="*** End Patch"),-1)
    if b==-1 or e==-1 or b>=e: raise ValueError("missing *** Begin Patch / *** End Patch markers")
    hunks=[]; i=b+1
    while i<e:
        ln=lines[i]
        if ln.startswith("*** Add File:"):
            fp=ln[len("*** Add File:"):].strip(); i+=1; c=""
            while i<e and not lines[i].startswith("***"):
                if lines[i].startswith("+"): c+=lines[i][1:]+"\n"
                i+=1
            if c.endswith("\n"): c=c[:-1]
            hunks.append(("add",fp,c,None))
        elif ln.startswith("*** Delete File:"):
            fp=ln[len("*** Delete File:"):].strip(); i+=1
            hunks.append(("delete",fp,None,None))
        elif ln.startswith("*** Update File:"):
            fp=ln[len("*** Update File:"):].strip(); i+=1; mv=None
            if i<e and lines[i].startswith("*** Move to:"):
                mv=lines[i][len("*** Move to:"):].strip(); i+=1
            chunks=[]; cur=None
            while i<e and not lines[i].startswith("***"):
                cl=lines[i]
                if cl.startswith("@@"):
                    cur={"ctx":cl[2:].strip() or None,"old":[],"new":[],"eof":False}
                    chunks.append(cur); i+=1; continue
                if cl=="*** End of File":
                    if cur is None: cur={"ctx":None,"old":[],"new":[],"eof":False}; chunks.append(cur)
                    cur["eof"]=True; i+=1; continue
                if cl[:1] in (" ","-","+"):
                    if cur is None: cur={"ctx":None,"old":[],"new":[],"eof":False}; chunks.append(cur)
                    if cl[0]==" ": cur["old"].append(cl[1:]); cur["new"].append(cl[1:])
                    elif cl[0]=="-": cur["old"].append(cl[1:])
                    else: cur["new"].append(cl[1:])
                    i+=1; continue
                i+=1
            hunks.append(("update",fp,mv,chunks))
        else:
            i+=1
    return hunks

def _abs(fp):
    return fp if fp.startswith("/") else "/testbed/"+fp

try:
    hunks=_parse(text)
except Exception as ex:
    print("ERROR parse: "+str(ex)); sys.exit()
if not hunks:
    print("ERROR: patch contains no file sections"); sys.exit()

pending={}; deletes=[]
def _readnow(p):
    if p in pending: return pending[p]
    return open(p,encoding="utf-8").read()

for h in hunks:
    typ=h[0]; tgt=_abs(h[1])
    if typ=="add":
        if os.path.exists(tgt) or tgt in pending:
            print("ERROR: Add File already exists: "+h[1]); sys.exit()
        c=h[2]; pending[tgt]= c if (c=="" or c.endswith("\n")) else c+"\n"
    elif typ=="delete":
        if not (os.path.exists(tgt) or tgt in pending):
            print("ERROR: Delete File not found: "+h[1]); sys.exit()
        pending.pop(tgt,None); deletes.append(tgt)
    else:
        mv=h[2]; chunks=h[3]
        if not (os.path.exists(tgt) or tgt in pending):
            print("ERROR: Update File not found: "+h[1]); sys.exit()
        try:
            new=_derive(_readnow(tgt),chunks)
        except Exception as ex:
            print("ERROR applying to "+h[1]+": "+str(ex)); sys.exit()
        if mv:
            dst=_abs(mv); deletes.append(tgt); pending.pop(tgt,None); pending[dst]=new
        else:
            pending[tgt]=new

for p in deletes:
    if os.path.exists(p):
        try: os.remove(p)
        except Exception: pass
warns=[]; written=0
for p,content in pending.items():
    d=os.path.dirname(p)
    if d and not os.path.exists(d): os.makedirs(d,exist_ok=True)
    open(p,"w",encoding="utf-8").write(content); written+=1
    if p.endswith(".py"):
        try: compile(content,p,"exec")
        except SyntaxError as ex: warns.append("WARNING: "+p+" has a syntax error after patch (fix it): "+str(ex))
print("ok: applied patch ("+str(written)+" file(s) written, "+str(len(deletes))+" deleted)")
for w in warns: print(w)
"""


def tool_apply_patch(cid, args):
    """Apply a Codex-format multi-file patch envelope inside the container.

    Accepts the OpenAI apply_patch language (``*** Begin Patch`` ... ``*** End
    Patch`` with ``*** Add/Update/Delete File:`` sections and ``@@``/``+``/``-``
    hunks). The whole envelope travels as base64 and is parsed + applied by the
    ported MiMo logic in python. Parsing/derivation runs before any write, so a bad
    hunk leaves the working tree untouched. Available to every config so the tool set
    stays identical across models.
    """
    patchtext = args.get("patchText") or args.get("patch") or args.get("input")
    if not patchtext:
        return ("ERROR: apply_patch requires 'patchText' (a '*** Begin Patch' ... "
                "'*** End Patch' envelope)")
    py = (
        "import base64,sys,os,re\n"
        f"text=base64.b64decode('{_b64(patchtext)}').decode('utf-8')\n"
        + _APPLY_PATCH_BODY
    )
    script = "cd /testbed && python - <<'PYEOF'\n" + py + "PYEOF\n"
    rc, out = exec_in(cid, script)
    return out.strip()[:1500] or "(no output)"


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
    "edit": tool_edit, "write": tool_write, "run": tool_run,
    "apply_patch": tool_apply_patch,
}


# Agent loop.
SYSTEM = (
    "You are an autonomous software engineer fixing a GitHub issue in a Python "
    "repository checked out at /testbed. You work by calling ONE tool per turn.\n\n"
    "Reply with EXACTLY one ```json fenced block and nothing else, of the form:\n"
    '```json\n{\"thought\": \"...\", \"tool\": \"<name>\", \"args\": {...}}\n```\n\n'
    "Tools:\n"
    "- ls   {\"path\": \"dir\"}                       list a directory\n"
    "- read {\"path\": \"f.py\", \"start\": 1, \"lines\": 200}  read file lines (numbered)\n"
    "- grep {\"pattern\": \"text\", \"path\": \".\"}      fixed-string search\n"
    "- edit {\"path\": \"f.py\", \"old\": \"exact snippet\", \"new\": \"replacement\"}  replace ONE exact snippet\n"
    "- apply_patch {\"patchText\": \"*** Begin Patch\\n*** Update File: f.py\\n@@\\n-old line\\n+new line\\n*** End Patch\"}  apply a Codex-format multi-file/multi-hunk patch\n"
    "- write{\"path\": \"new.py\", \"content\": \"file contents\"}  create a NEW file\n"
    "- run  {\"cmd\": \"python -m pytest path::test -q\"}  run a shell command (180s cap)\n"
    "- finish {\"reason\": \"...\"}                    stop; your git diff is the patch\n\n"
    "Guidance:\n"
    "- Explore with grep/read first to locate the real cause; don't guess.\n"
    "- `python` and `pytest` already point to the project's env. cwd is /testbed.\n"
    "- Reproduce the bug, make the MINIMAL fix, then re-run the relevant tests.\n"
    "- Edit existing files with `edit` (give a unique `old` snippet; exact whitespace "
    "preferred but a fuzzy fallback locates near-misses); `write` only creates NEW "
    "files. Never paste a whole file back.\n"
    "- For larger or multi-file changes you may use `apply_patch` (Codex patch format) "
    "instead of several `edit` calls; context/`-` lines need not match whitespace exactly.\n"
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


WINDOW_TURNS = 8  # context bound: turns of (assistant+observation) kept in the prompt


def trim_messages(messages, keep=WINDOW_TURNS):
    """Bound the prompt to system + initial issue + the last `keep` turns.

    chat-completions is stateless, so the client must resend context every turn; but
    resending the WHOLE un-trimmed transcript makes prompt tokens grow ~O(N^2) over a
    run and is what blows up long agent loops (the dominant cost in the failing
    cases). A fixed recent window makes it ~O(N*K). messages[0:2] (system + the
    original issue) are always kept so the task is never forgotten; older turns are
    replaced by a single stub. Real clients do richer compaction/repo-maps -- this is
    a deliberately simple, uniform version applied to every config.
    """
    head = messages[:2]
    rest = messages[2:]
    tail_n = keep * 2  # each turn contributes an assistant + a user(observation) msg
    if len(rest) <= tail_n:
        return messages
    elided = len(rest) - tail_n
    stub = {"role": "user", "content":
            f"[{elided} earlier transcript messages elided to bound context. The issue "
            "above and your most recent steps below are retained; re-read files if unsure.]"}
    return head + [stub] + rest[-tail_n:]


# Max mode (step-level ensemble): N propose-only candidates -> judge -> winner.
# Inspired by MiMo-Code's experimental "max" agent. At each step we sample N
# candidate next-actions from the SAME model (raised temperature for diversity),
# a judge picks the single best, and ONLY that one is executed. Unlike a terminal-
# level fusion this keeps ONE coherent trajectory: the loser drafts and the judge
# are pure overhead that never enter the transcript (so context grows by the winner
# alone). Diversity here comes from sampling, so candidate_temperature must be > 0:
# at temperature 0 the draws are identical and max mode degenerates to a single
# call plus wasted judge spend. Enabled per-config via a "max_mode" key:
#   { "name": "maxstep-...", "model": "...", "max_mode": {"candidates": 5,
#     "candidate_temperature": 0.7, "judge": "<optional judge model>"} }
MAX_MODE_JUDGE_SYSTEM = (
    "You are a judge selecting the single best NEXT step for an autonomous coding "
    "agent fixing a bug. You will see several independent candidate next steps for "
    "the SAME state; each is one tool call (thought + tool + args). Pick the ONE "
    "that is most correct, grounded, and useful. Prefer a step that locates the real "
    "cause or VERIFIES the fix by running the actual failing tests over one that "
    "finishes prematurely or guesses. Reply with ONLY the integer index."
)


def _render_candidate(action, label):
    a = action or {}
    args = json.dumps(a.get("args", {}), ensure_ascii=False)
    if len(args) > 600:
        args = args[:600] + " ...[truncated]"
    return (f"### Candidate {label}\n"
            f"thought: {(a.get('thought') or '').strip()[:400]}\n"
            f"tool: {a.get('tool')}\n"
            f"args: {args}")


def judge_pick(cfg, candidates, timeout, acc):
    """Index of the winning candidate. Defaults to 0 on any parse/range/error issue,
    so a flaky judge never blocks the step."""
    if len(candidates) == 1:
        return 0
    mm = cfg.get("max_mode") or {}
    judge_model = mm.get("judge") or cfg["model"]
    rendered = "\n\n".join(_render_candidate(c["action"], i) for i, c in enumerate(candidates))
    prompt = (f"There are {len(candidates)} candidate next steps, indexed "
              f"0..{len(candidates) - 1}.\n\n{rendered}\n\n"
              f"Reply with ONLY the integer index (0..{len(candidates) - 1}) of the best one.")
    jcfg = {k: v for k, v in cfg.items() if k != "max_mode"}
    jcfg = {**jcfg, "model": judge_model, "temperature": 0}
    try:
        content, usage = call_model(jcfg, [
            {"role": "system", "content": MAX_MODE_JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ], timeout=timeout)
    except Exception as e:
        print(f"      judge error {e}; defaulting to candidate 0")
        return 0
    accumulate_usage(acc, usage)
    m = re.search(r"\d+", content or "")
    if not m:
        return 0
    pick = int(m.group(0))
    return pick if 0 <= pick < len(candidates) else 0


def run_max_step(cfg, messages, timeout, acc):
    """One max-mode step. Returns (winner_content, winner_action), or None to signal
    the caller to fall back to a single normal call (when every candidate failed to
    produce a parseable action). All candidate + judge tokens are accumulated into
    `acc` so cost reflects the true ~Nx spend; only the winner enters the transcript."""
    mm = cfg.get("max_mode") or {}
    n = max(1, int(mm.get("candidates", 5)))
    ctemp = mm.get("candidate_temperature", 0.7)
    ccfg = {k: v for k, v in cfg.items() if k != "max_mode"}
    ccfg = {**ccfg, "temperature": ctemp}
    trimmed = trim_messages(messages)
    candidates = []
    for i in range(n):
        try:
            content, usage = call_model(ccfg, trimmed, timeout=timeout)
        except Exception as e:
            print(f"      candidate {i}: model error {e}")
            continue
        accumulate_usage(acc, usage)  # losers are real spend (billing / overhead)
        action, err = parse_action(content)
        if err:
            continue
        candidates.append({"content": content, "action": action})
    if not candidates:
        return None
    pick = judge_pick(cfg, candidates, timeout, acc)
    win = candidates[pick]
    print(f"      max-step: {len(candidates)}/{n} survivors, judge -> {pick} "
          f"({(win['action'] or {}).get('tool')})")
    return win["content"], win["action"]


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
        max_mode = cfg.get("max_mode")
        for turns in range(1, max_turns + 1):
            if max_mode:
                step = run_max_step(cfg, messages, turn_timeout, acc)
                if step is None:
                    # every candidate failed: degrade to a single normal call so the
                    # step still makes progress.
                    try:
                        content, usage = call_model(cfg, trim_messages(messages), timeout=turn_timeout)
                    except Exception as e:
                        print(f"    turn {turns}: model error {e}")
                        break
                    accumulate_usage(acc, usage)
                    action, err = parse_action(content)
                else:
                    content, action, err = step[0], step[1], None
            else:
                try:
                    content, usage = call_model(cfg, trim_messages(messages), timeout=turn_timeout)
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
                            "apply the fix now with `edit`, run the test to confirm, then "
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
