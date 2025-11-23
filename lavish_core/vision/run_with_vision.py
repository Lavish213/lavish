#!/usr/bin/env python3
"""Force-enable Vision exactly once and exec the Lavish runner."""

from __future__ import annotations
import os, sys, shlex

# ---- hard guards against loops ------------------------------------------------
if os.environ.get("LAVISH_LAUNCHED") == "1":
    # We're already inside the real runner; do nothing.
    print("⚠️ Already inside runner (LAVISH_LAUNCHED=1) — not re-launching.")
    sys.exit(0)

# Ensure .env is optional
try:
    from pathlib import Path
    from dotenv import load_dotenv  # type: ignore
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    pass

# Force env for any code that reads it at import-time
os.environ["FORCE_ENABLE_VISION"] = "1"
os.environ["ENABLE_VISION"] = "1"
os.environ["LAVISH_LAUNCHED"] = "1"  # sentinel to stop any relaunch attempts

# Build argv for runner module (no subprocess, no duplication)
def _build_argv() -> list[str]:
    argv = [sys.executable, "-m", "lavish_core.run"]
    # mode: CLI --mode X OR env MODE else 'dev'
    mode = None
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        mode = sys.argv[i+1] if i+1 < len(sys.argv) else None
    if mode is None:
        mode = os.getenv("MODE", "dev")
    argv.append(mode)

    # always pass single --enable-vision
    argv.append("--enable-vision")

    # symbols: from CLI or env
    syms = None
    if "--symbols" in sys.argv:
        j = sys.argv.index("--symbols")
        syms = sys.argv[j+1] if j+1 < len(sys.argv) else None
    syms = syms or os.getenv("SYMBOLS") or os.getenv("SYMBOLS_LIST") or ""
    if syms:
        argv += ["--symbols", syms]

    # pass through any extra args except our own
    skip = {"run_with_vision.py", "--mode", "--symbols"}
    k = 0
    while k < len(sys.argv):
        tok = sys.argv[k]
        nxt = sys.argv[k+1] if k+1 < len(sys.argv) else None
        if tok in skip or tok.endswith("run_with_vision.py"):
            k += 1
            continue
        if tok in ("--mode", "--symbols"):
            k += 2
            continue
        if tok not in ("-m",):
            argv.append(tok)
        k += 1
    return argv

argv = _build_argv()
print("▶️ Launching Lavish with Vision ON:")
print("   " + " ".join(shlex.quote(a) for a in argv))

# Replace current process with the runner (no child process, no loops)
os.execv(sys.executable, argv)