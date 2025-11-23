# fix_datetimes.py
# Refactors deprecated datetime.now(timezone.utc) usages to timezone-aware forms.
# Default TZ: America/Los_Angeles (PST/PDT). Use --tz utc for UTC output.

from __future__ import annotations
import argparse, re, sys
from pathlib import Path

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="Fix datetime.now(timezone.utc) -> tz-aware now()")
ap.add_argument("--root", default=".", help="Project root to scan")
ap.add_argument("--tz", choices=["local", "utc"], default="local",
                help="local = America/Los_Angeles, utc = timezone.utc")
ap.add_argument("--zone", default="America/Los_Angeles",
                help="IANA zone when --tz local (default America/Los_Angeles)")
ap.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
ap.add_argument("--include-ext", nargs="*", default=[".py"], help="File extensions to process")
ap.add_argument("--exclude", nargs="*", default=[".git", "venv", ".venv", "__pycache__"],
                help="Directories to skip")
args = ap.parse_args()

ROOT = Path(args.root).resolve()
exts = set(e.lower() for e in args.include_ext)
skip = set(args.exclude)

# ---------- Replacement targets ----------
# Patterns we’ll fix (module-qualified variants too)
PATTERNS = [
    # utcnow().isoformat() + 'Z'
    (re.compile(r"datetime\.utcnow\(\)\.isoformat\(\)\s*\+\s*['\"]Z['\"]"), "datetime.now({TZ}).isoformat()"),
    (re.compile(r"dt\.datetime\.utcnow\(\)\.isoformat\(\)\s*\+\s*['\"]Z['\"]"), "dt.datetime.now({DT_TZ}).isoformat()"),

    # bare utcnow().isoformat()
    (re.compile(r"datetime\.utcnow\(\)\.isoformat\(\)"), "datetime.now({TZ}).isoformat()"),
    (re.compile(r"dt\.datetime\.utcnow\(\)\.isoformat\(\)"), "dt.datetime.now({DT_TZ}).isoformat()"),

    # bare utcnow()
    (re.compile(r"datetime\.utcnow\(\)"), "datetime.now({TZ})"),
    (re.compile(r"dt\.datetime\.utcnow\(\)"), "dt.datetime.now({DT_TZ})"),
]

def should_skip(path: Path) -> bool:
    parts = set(p.name for p in path.parents)
    if any(x in skip for x in parts): return True
    if any(x in path.parts for x in skip): return True
    return False

def ensure_imports(code: str, use_local: bool, zone: str) -> str:
    """Add timezone/ZoneInfo imports if needed for 'from datetime import datetime' style.""", timezone
    changed = code

    # When using 'datetime.now(timezone.utc)' we need 'timezone'
    if not use_local:
        # If file has 'from datetime import datetime' but not 'timezone', add it, timezone
        pattern = re.compile(r"from\s+datetime\s+import\s+([^\n#]+)")
        def _add_timezone(m):
            items = [i.strip() for i in m.group(1).split(",")]
            if "timezone" not in [i.split(" as ")[0] for i in items]:
                items.append("timezone")
                return f"from datetime import {', '.join(items)}", timezone
            return m.group(0)
        changed = pattern.sub(_add_timezone, changed, count=0)
    else:
        # Local zone uses ZoneInfo import
        if "ZoneInfo" not in changed:
            # Insert after first import block
            lines = changed.splitlines()
            insert_at = 0
            for i, line in enumerate(lines[:20]):
                if line.startswith("from ") or line.startswith("import "):
                    insert_at = i + 1
            lines.insert(insert_at, "from zoneinfo import ZoneInfo")
            changed = "\n".join(lines)

    return changed

def apply_replacements(text: str, use_local: bool, zone: str) -> tuple[str, bool]:
    TZ = "ZoneInfo(%r)" % zone if use_local else "timezone.utc"
    DT_TZ = "ZoneInfo(%r)" % zone if use_local else "dt.timezone.utc"

    new = text
    touched = False
    for pat, repl in PATTERNS:
        replaced = pat.sub(repl.format(TZ=TZ, DT_TZ=DT_TZ), new)
        if replaced != new:
            touched = True
            new = replaced
    if touched:
        new = ensure_imports(new, use_local, zone)
    return new, touched

def process_file(path: Path) -> tuple[int, int]:
    src = path.read_text(encoding="utf-8")
    new, touched = apply_replacements(src, args.tz == "local", args.zone)
    if touched and not args.dry_run:
        # backup
        path.with_suffix(path.suffix + ".bak").write_text(src, encoding="utf-8")
        path.write_text(new, encoding="utf-8")
    return (1 if touched else 0), (0 if args.dry_run else (1 if touched else 0))

def main():
    changed_files = 0
    written_files = 0
    for p in ROOT.rglob("*"):
        if p.is_dir(): continue
        if should_skip(p): continue
        if p.suffix.lower() not in exts: continue
        try:
            c, w = process_file(p)
            changed_files += c
            written_files += w
        except Exception as e:
            print(f"⚠️  Failed {p}: {e}", file=sys.stderr)
    mode = f"LOCAL({args.zone})" if args.tz == "local" else "UTC"
    print(f"✅ Done. Matches changed: {changed_files} | Files written: {written_files} | Mode: {mode}")

if __name__ == "__main__":
    main()