#!/usr/bin/env python3
"""
Super Bot Injection — Lavish Wall Street Edition
-----------------------------------------------
One-shot bootstrapper to clone/update many GitHub repos, lay down import-safe wrappers,
and prep dependencies & notes. Designed to be idempotent and safe.

Usage:
  # Preview only:
  python super_injector.py --dry-run

  # Apply (clone/update + wrappers):
  python super_injector.py --apply

  # Apply + install deps:
  python super_injector.py --apply --install-deps

  # Advanced:
  python super_injector.py --apply --workers 8 --shallow --categories agents utils fastapi bots scrapers
  python super_injector.py --apply --repos-file repos.txt
  python super_injector.py --apply --yaml super_injector.yaml
"""

from __future__ import annotations
import os, sys, subprocess, shlex, json, textwrap, argparse, shutil, re, concurrent.futures
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXT_DIR = ROOT / "external_repos"
WRAP_DIR = ROOT / "lavish_core" / "external"
README = ROOT / "super_injector.README"
LOG = ROOT / "super_injector.log"
REQ = ROOT / "requirements-lavish.txt"

DEFAULT_CATEGORIES = ["agents", "utils", "fastapi", "bots", "scrapers", "vision", "vectors", "trading"]

# ---- Built-in seed repos (safe & popular). You can add hundreds via repos.txt or YAML. ----
BUILTIN = {
    "vision": [
        "https://github.com/openai/CLIP.git",
        "https://github.com/salesforce/BLIP.git",
        "https://github.com/PaddlePaddle/PaddleOCR.git",
        "https://github.com/JaidedAI/EasyOCR.git",
        "https://github.com/huggingface/transformers.git",
        "https://github.com/ultralytics/ultralytics.git",
    ],
    "vectors": [
        "https://github.com/chroma-core/chromadb.git",
        "https://github.com/facebookresearch/faiss.git",
        "https://github.com/UKPLab/sentence-transformers.git",
    ],
    "agents": [
        "https://github.com/langchain-ai/langchain.git",
        "https://github.com/hwchase17/langchain.git",  # alt mirror naming
        "https://github.com/yoheinakajima/awesome-agi.git",  # reference lists
    ],
    "utils": [
        "https://github.com/psf/requests.git",
        "https://github.com/pallets/click.git",
        "https://github.com/PyCQA/ruff.git",
    ],
    "fastapi": [
        "https://github.com/tiangolo/fastapi.git",
        "https://github.com/encode/uvicorn.git",
        "https://github.com/encode/starlette.git",
    ],
    "bots": [
        "https://github.com/python-telegram-bot/python-telegram-bot.git",
        "https://github.com/discordjs/discord.js.git",  # JS reference
        "https://github.com/aiogram/aiogram.git",
    ],
    "scrapers": [
        "https://github.com/codelucas/newspaper.git",
        "https://github.com/fhamborg/news-please.git",
        "https://github.com/justinwilaby/yfinance-news.git",  # small ref
    ],
    "trading": [
        "https://github.com/alpacahq/alpaca-trade-api-python.git",
        "https://github.com/finnhubio/finnhub-python.git",
        "https://github.com/ranaroussi/yfinance.git",
    ],
}

# Suggested pip deps (deduped into requirements-lavish.txt)
PIP_SEEDS = [
    "torch", "transformers", "sentence-transformers",
    "pillow", "pandas", "numpy", "faiss-cpu", "chromadb", "langchain",
    "paddleocr", "easyocr", "opencv-python",
    "fastapi", "uvicorn", "starlette",
    "requests", "python-dotenv", "yfinance",
    "alpaca-trade-api", "finnhub-python",
]

def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
    print(msg)

def parse_args():
    ap = argparse.ArgumentParser(description="Lavish Super Bot Injection")
    ap.add_argument("--apply", action="store_true", help="Perform changes (otherwise dry-run).")
    ap.add_argument("--install-deps", action="store_true", help="pip install requirements-lavish.txt after generation.")
    ap.add_argument("--workers", type=int, default=6, help="Parallel clone workers.")
    ap.add_argument("--shallow", action="store_true", help="Use shallow git clone (depth=1).")
    ap.add_argument("--repos-file", type=str, default="repos.txt", help="Plain text list of repos to include.")
    ap.add_argument("--yaml", type=str, default="super_injector.yaml", help="YAML config with categories -> repo list.")
    ap.add_argument("--categories", nargs="*", default=[], help="Limit to these categories (else all).")
    ap.add_argument("--dry-run", action="store_true", help="Force dry-run (overrides --apply).")
    return ap.parse_args()

def read_repos_file(path: Path) -> list[str]:
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if not line.endswith(".git") and re.match(r"^[\w\-]+\/[\w\-.]+$", line):
            out.append(f"https://github.com/{line}.git")
        else:
            out.append(line)
    return out

def read_yaml_map(path: Path) -> dict[str, list[str]]:
    # Optional dependency-free YAML reader for simple lists (very limited)
    if not path.exists(): return {}
    raw = path.read_text(encoding="utf-8")
    groups: dict[str, list[str]] = {}
    cur = None
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"): continue
        if re.match(r"^[\w\-]+:\s*$", s):
            cur = s.split(":")[0].strip()
            groups[cur] = []
        elif s.startswith("- ") and cur:
            val = s[2:].strip()
            if not val.endswith(".git") and re.match(r"^[\w\-]+\/[\w\-.]+$", val):
                val = f"https://github.com/{val}.git"
            groups[cur].append(val)
    return groups

def git_clone_or_update(url: str, dst: Path, shallow: bool, apply: bool) -> str:
    if dst.exists():
        cmd = f"git -C {shlex.quote(str(dst))} pull --ff-only"
    else:
        depth = "--depth 1" if shallow else ""
        cmd = f"git clone {depth} {shlex.quote(url)} {shlex.quote(str(dst))}"
    if not apply:
        return f"(dry-run) {cmd}"
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"OK: {cmd}"
    except subprocess.CalledProcessError as e:
        return f"ERR: {cmd}\n{e.stderr.decode(errors='ignore')}"

def slug_from_url(url: str) -> str:
    base = url.rstrip("/").split("/")[-1]
    if base.endswith(".git"): base = base[:-4]
    return base.lower().replace(".", "_").replace("-", "_")

def write_wrapper(module: str, repo_dir: Path, apply: bool):
    WRAP_DIR.mkdir(parents=True, exist_ok=True)
    (WRAP_DIR / "__init__.py").write_text("# auto-generated\n", encoding="utf-8") if apply else None
    code = f'''"""
Lavish External Wrapper — {module}
Import-safe adapter for external repo: {repo_dir.name}
"""

from __future__ import annotations

try:
    # Try to import if the real package is installed/available
    import sys, import importlib, pathlib
    _repo = pathlib.Path({repr(str(repo_dir))})
    if _repo.exists():
        sys.path.insert(0, str(_repo))
    # Example: detect a python package inside the repo
    # _pkg = importlib.import_module("{module}")
except Exception as e:
    _pkg = None  # still keep wrapper importable

def available() -> bool:
    try:
        import importlib
        return True
    except Exception:
        return False

def info() -> dict:
    return {{
        "module": {repr(module)},
        "repo_dir": {repr(str(repo_dir))},
        "status": "available" if available() else "missing",
    }}
'''
    out = WRAP_DIR / f"{module}.py"
    if apply:
        out.write_text(code, encoding="utf-8")
    return out

def upsert_requirements(pkgs: list[str], apply: bool):
    existing = set()
    if REQ.exists():
        for ln in REQ.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                existing.add(ln)
    final = sorted(existing | set(pkgs))
    if apply:
        REQ.write_text("\n".join(final) + "\n", encoding="utf-8")
    return final

def ensure_readme(repos: dict[str, list[str]], apply: bool):
    txt = ["# Lavish Super Injector — What was done",
           "",
           "## Repos cloned/updated (by category)",
           ""]
    for cat, lst in repos.items():
        txt.append(f"### {cat}")
        for u in lst:
            txt.append(f"- {u}")
        txt.append("")
    txt.append("## Next steps")
    txt.append("- Create/activate a venv")
    txt.append("- pip install -r requirements-lavish.txt")
    txt.append("- Replace wrapper TODOs with direct imports as you adopt each lib")
    content = "\n".join(txt)
    if apply:
        README.write_text(content, encoding="utf-8")
    return content

def main():
    args = parse_args()
    apply = args.apply and not args.dry_run
    shallow = args.shallow
    workers = max(1, args.workers)

    # 1) gather repos from: YAML (categories), repos.txt (flat), builtin seeds
    yaml_map = read_yaml_map(ROOT / args.yaml)
    repos_by_cat: dict[str, list[str]] = {}

    categories = args.categories or list(set(DEFAULT_CATEGORIES + list(yaml_map.keys())))
    for cat in categories:
        seed = BUILTIN.get(cat, [])
        ext = yaml_map.get(cat, [])
        repos_by_cat[cat] = list(dict.fromkeys(seed + ext))  # dedupe preserve order

    # Flat repos file goes into a "custom" bucket
    extras = read_repos_file(ROOT / args.repos_file)
    if extras:
        repos_by_cat["custom"] = list(dict.fromkeys(extras))

    # 2) clone/update in parallel
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for cat, lst in repos_by_cat.items():
        (EXT_DIR / cat).mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {}
        for cat, lst in repos_by_cat.items():
            for url in lst:
                slug = slug_from_url(url)
                dst = EXT_DIR / cat / slug
                future = ex.submit(git_clone_or_update, url, dst, shallow, apply)
                future_map[future] = (cat, url, dst, slug)
        for fut in concurrent.futures.as_completed(future_map):
            cat, url, dst, slug = future_map[fut]
            res = fut.result()
            log(f"[{cat}] {res}")

    # 3) wrappers for each repo
    for cat, lst in repos_by_cat.items():
        for url in lst:
            slug = slug_from_url(url)
            repo_dir = EXT_DIR / cat / slug
            wrap_path = write_wrapper(slug, repo_dir, apply)
            log(f"[wrapper] {wrap_path}")

    # 4) requirements
    final_reqs = upsert_requirements(PIP_SEEDS, apply)
    log(f"[requirements] {len(final_reqs)} packages in {REQ.name}")

    # 5) readme
    ensure_readme(repos_by_cat, apply)
    log("[done] Super injection complete." if apply else "[dry-run] No changes written.")

if __name__ == "__main__":
    main()