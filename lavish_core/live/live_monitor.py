from __future__ import annotations
import os, sys, time, json, traceback, threading, concurrent.futures, hashlib
from pathlib import Path
from typing import Set, Dict, Any, Iterable


from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lavish_core.vision.candle_reader import extract_candles  # ✅ your API
# ── Ensure project root on path ───────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT not in sys.path:
    sys.path.append(ROOT)


# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR       = Path("data/images_in").resolve()
PROCESSED_DIR  = Path("data/images_done").resolve()
ERROR_DIR      = Path("data/images_err").resolve()
SIDECAR_DIR    = Path("data/analysis_out").resolve()  # JSON summaries per image
IMAGE_EXTS     = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
POLL_SEC       = 2.0
MAX_WORKERS    = max(2, os.cpu_count() or 4)  # parallel parse
STABILITY_WAIT = 0.6  # seconds to re-check file size (avoid half-written files)

for d in (PROCESSED_DIR, ERROR_DIR, SIDECAR_DIR):
    d.mkdir(parents=True, exist_ok=True)

console = Console()
_stop_flag = False
_seen_fingerprints: Set[str] = set()  # dedupe across restarts in same run


# ── Helpers ──────────────────────────────────────────────────────────────────
def _fingerprint(p: Path) -> str:
    """Stable-ish identity for a file (name + size + mtime)."""
    try:
        st = p.stat()
        h = hashlib.sha1(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
        return h
    except Exception:
        return p.name

def _is_stable_file(p: Path) -> bool:
    """Avoid processing while a screenshot is still being written to disk."""
    try:
        s1 = p.stat().st_size
        time.sleep(STABILITY_WAIT)
        s2 = p.stat().st_size
        return s1 == s2 and s1 > 0
    except FileNotFoundError:
        return False

def _list_images(folder: Path) -> Iterable[Path]:
    for p in folder.glob("*"):
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
            yield p

def _write_sidecar(image: Path, payload: Dict[str, Any]) -> Path:
    out = SIDECAR_DIR / f"{image.stem}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out


# ── Summaries + Hooks ────────────────────────────────────────────────────────
def _summarize(image_path: Path, result: Dict[str, Any], grade: Dict[str, Any] | None, probs: Dict[str, Any] | None) -> None:
    bars = result.get("bars", [])
    theme = result.get("theme")
    overlay = result.get("debug_overlay", "-")

    t = Table(title=f"{image_path.name}", show_header=True, header_style="bold cyan")
    t.add_column("Metric", justify="right", style="cyan")
    t.add_column("Value", style="yellow")
    t.add_row("Bars Parsed", str(len(bars)))
    t.add_row("Theme", str(theme))
    t.add_row("Overlay", str(overlay))

    if grade:
        t.add_row("Grade", f"{grade.get('score','?')} ({grade.get('label','-')})")
    if probs:
        t.add_row("Predictor", json.dumps(probs))

    console.print(Panel(t, title="[magenta]Lavish Candle Reader[/magenta]"))

    # Breakdown table (optional)
    if grade and isinstance(grade.get("breakdown"), dict):
        bd = Table(title="Grading Breakdown")
        bd.add_column("Factor", style="green")
        bd.add_column("Score", style="white")
        for k, v in grade["breakdown"].items():
            bd.add_row(str(k), str(v))
        console.print(bd)


def _alert_hook(image: Path, grade: Dict[str, Any] | None, probs: Dict[str, Any] | None) -> None:
    """
    Plug your notification here (Discord/Slack/Webhook).
    This is a stub; implement as needed. Called after each successful parse.
    """
    try:
        # Example (disabled):
        # if grade and grade.get("score", 0) >= 80:
        #     send_discord(f"🔥 Strong pattern in {image.name}: {grade['label']} ({grade['score']})")
        pass
    except Exception:
        pass


# ── One-file processor (runs in pool) ────────────────────────────────────────
def _process_image(p: Path) -> None:
    fp = _fingerprint(p)
    if fp in _seen_fingerprints:
        return
    if not _is_stable_file(p):
        return  # try next poll

    try:
        result = extract_candles(p)  # ✅ your reader
        bars = result.get("bars", [])

        # Optional grading
        grade = None
        try:
            from lavish_core.analysis.grading_engine import grade_signal
            grade = grade_signal(bars)
        except Exception as e:
            console.print(f"[yellow]Grading skipped: {e}[/yellow]")

        # Optional predictor
        probs = None
        try:
            from lavish_core.ai.predictor import predict_outcome
            probs = predict_outcome(bars)
        except Exception as e:
            console.print(f"[yellow]Predictor skipped: {e}[/yellow]")

        # Persist sidecar JSON
        sidecar = {
            "image": str(p),
            "summary": {
                "theme": result.get("theme"),
                "bars_parsed": len(bars),
                "overlay": result.get("debug_overlay"),
            },
            "grade": grade,
            "predictor": probs,
            "raw": result,  # keep if you want full payloads for replay engine
        }
        _write_sidecar(p, sidecar)

        # Pretty print
        _summarize(p, result, grade, probs)

        # Move file → processed (archive)
        dst = PROCESSED_DIR / p.name
        try:
            p.rename(dst)
        except Exception:
            # cross-device or busy → copy then remove
            dst.write_bytes(p.read_bytes())
            p.unlink(missing_ok=True)

        _seen_fingerprints.add(fp)
        _alert_hook(dst, grade, probs)

    except Exception as e:
        console.print(f"[red]Error on {p.name}: {e}[/red]")
        console.print(traceback.format_exc())
        # Archive into errors for forensics
        try:
            err_dst = ERROR_DIR / p.name
            if p.exists():
                p.rename(err_dst)
        except Exception:
            pass


# ── Watcher (daemon thread) ─────────────────────────────────────────────────
def watch_loop(poll_sec: float = POLL_SEC) -> None:
    console.print("[bold cyan]🚀 Live monitor started. Drop chart screenshots into data/images_in[/bold cyan]")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures: Set[concurrent.futures.Future] = set()
        while not _stop_flag:
            try:
                for p in _list_images(DATA_DIR):
                    # Submit only if we haven't seen/queued it yet
                    fp = _fingerprint(p)
                    if fp in _seen_fingerprints:
                        continue
                    # one task per image; processor re-checks stability
                    futures.add(pool.submit(_process_image, p))

                # Reap finished tasks to keep set small
                done = {f for f in futures if f.done()}
                for f in done:
                    try: _ = f.result()
                    except Exception: pass
                futures -= done

            except Exception as e:
                console.print(f"[red]Watcher error: {e}[/red]")
            time.sleep(poll_sec)

    console.print("[yellow]👋 Live monitor stopped gracefully.[/yellow]")


def start_watcher():
    """Start the watcher in a daemon thread (safe with asyncio/uvloop)."""
    global _stop_flag
    _stop_flag = False
    t = threading.Thread(target=watch_loop, daemon=True)
    t.start()
    return t


def stop_watcher():
    """Signal the watcher to stop gracefully."""
    global _stop_flag
    _stop_flag = True

