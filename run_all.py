#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lavish — run_all.py (LAN-ready, dual-logging, supervised)
Drop in repo root and run:
  python run_all.py --mode dev --symbols NVDA,AAPL --enable-vision --watch
"""

from __future__ import annotations
import os, sys, asyncio, logging, contextlib, time, json, random, signal, textwrap
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# ───────────────────────── Path & ENV ─────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logging.handlers import RotatingFileHandler

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_a, **_k): pass

load_dotenv(ROOT / ".env")

# ─────────────────── Configuration (from ENV) ─────────────────
MODE = os.getenv("MODE", "dev").lower()
ALPACA_BASE_URL = (os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
TIMEZONE = os.getenv("TIMEZONE", "America/Los_Angeles")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REFRESH_MIN = int(os.getenv("BRAIN_REFRESH_MIN", os.getenv("BOT_REFRESH_INTERVAL_MIN", "15")))
HEALTH_HOST = os.getenv("HEALTH_HOST", "0.0.0.0")
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8000"))

# Optional toggles
ENABLE_VISION = os.getenv("ENABLE_VISION", "0") in ("1", "true", "yes")
ENABLE_WATCH  = os.getenv("ENABLE_WATCH", "0") in ("1", "true", "yes")

# Symbols (CLI can override)
DEFAULT_SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,NVDA,SPY,QQQ").split(",") if s.strip()]

# ───────────────────── Logging (dual-mode) ────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PLAIN_LOG = LOG_DIR / "run_all.log"
JSON_LOG  = LOG_DIR / "run_all.jsonl"

class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return ct.strftime(datefmt or "%Y-%m-%dT%H:%M:%S.%fZ")

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        # Attach extra fields if present
        for k, v in record.__dict__.items():
            if k not in ("msg","args","levelname","levelno","pathname","filename","module",
                         "exc_info","exc_text","stack_info","lineno","funcName","created",
                         "msecs","relativeCreated","thread","threadName","processName","process"):
                base[k] = v
        return json.dumps(base, ensure_ascii=False)

def build_logger() -> logging.Logger:
    logger = logging.getLogger("lavish.run_all")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.handlers.clear()

    fmt = UTCFormatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    # Console (clean)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logger.level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file (clean)
    fh = RotatingFileHandler(PLAIN_LOG, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logger.level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Rotating JSONL (structured)
    jh = RotatingFileHandler(JSON_LOG, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    jh.setLevel(logger.level)
    jh.setFormatter(JSONFormatter())
    logger.addHandler(jh)
    return logger

log = build_logger()

# ───────────────────── Secret masking helpers ─────────────────
SECRET_HINTS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "ACCESS", "API", "WEBHOOK", "CLIENT_ID",
    "CLIENT_SECRET", "AUTH", "BEARER"
)

def _mask(s: Optional[str]) -> str:
    if not s: return ""
    s = str(s)
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + "…" + s[-4:]

def env_snapshot(mask: bool = True) -> Dict[str, str]:
    out = {}
    for k, v in os.environ.items():
        if any(h in k.upper() for h in SECRET_HINTS):
            out[k] = _mask(v) if mask else v
        else:
            # keep short, avoid dumping massive values
            vv = (v or "")
            out[k] = vv if len(vv) < 256 else vv[:252] + "…"
    return out

def vet_env(required_keys: List[str]) -> Dict[str, Any]:
    issues = []
    for k in required_keys:
        if not os.getenv(k):
            issues.append(f"Missing {k}")
    ok = len(issues) == 0
    return {"ok": ok, "issues": issues}

# Critical (non-fatal) checks
REQUIRED_SUGGESTED = [
    "OPENAI_API_KEY",
    "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
    "NEWSAPI_KEY", "POLYGON_API_KEY", "FINNHUB_API_KEY",
]
vet = vet_env(REQUIRED_SUGGESTED)
if not vet["ok"]:
    log.warning("ENV vetting warnings: %s", vet["issues"])

# ───────────────────────── Health API ─────────────────────────
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

app = FastAPI(title="Lavish Health", version="1.0.0")

BOOT_TS = time.time()
TASK_STATES: Dict[str, Dict[str, Any]] = {}

def set_task_state(name: str, **fields):
    st = TASK_STATES.setdefault(name, {})
    st.update(fields)
    st["updated"] = datetime.now(timezone.utc).isoformat()

@app.get("/status")
def status():
    uptime_s = int(time.time() - BOOT_TS)
    return {
        "mode": MODE,
        "alpaca_env": "paper" if "paper" in ALPACA_BASE_URL else "live",
        "timezone": TIMEZONE,
        "uptime_seconds": uptime_s,
        "refesh_min": REFRESH_MIN,
        "tasks": TASK_STATES,
        "env_vet": vet,
    }

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    # simple Prometheus-ish
    lines = [f"lavish_uptime_seconds {int(time.time()-BOOT_TS)}"]
    for name, st in TASK_STATES.items():
        lines.append(f'lavish_task_running{{task="{name}"}} {1 if st.get("running") else 0}')
        if "restarts" in st:
            lines.append(f'lavish_task_restarts{{task="{name}"}} {st.get("restarts",0)}')
    return "\n".join(lines) + "\n"

@app.get("/env/keys")
def env_keys():
    return env_snapshot(mask=True)

@app.get("/logs/tail", response_class=PlainTextResponse)
def logs_tail(n: int = 200):
    try:
        p = PLAIN_LOG
        if not p.exists():
            return "(no log file yet)"
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()[-max(1, n):]
        return "\n".join(lines) + "\n"
    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)

# ───────────────────── Supervisor primitives ─────────────────
class Supervisor:
    def __init__(self, name: str, factory, max_restarts: int = 8):
        self.name=name; self.factory=factory
        self.max=max_restarts; self.restarts=0
        self.task: Optional[asyncio.Task] = None
        self.stop_evt = asyncio.Event()

    async def start(self):
        set_task_state(self.name, running=True, restarts=self.restarts)
        while not self.stop_evt.is_set():
            try:
                log.info("▶️  start %s", self.name)
                self.task = asyncio.create_task(self.factory())
                await self.task
                if self.stop_evt.is_set(): break
                log.info("✅ %s exited cleanly", self.name)
                set_task_state(self.name, running=False)
                break
            except asyncio.CancelledError:
                log.info("⛔ %s cancelled", self.name)
                set_task_state(self.name, running=False, cancelled=True)
                break
            except Exception as e:
                self.restarts += 1
                set_task_state(self.name, running=False, restarts=self.restarts, last_error=str(e))
                if self.restarts > self.max:
                    log.error("❌ %s exceeded max restarts (%d): %s", self.name, self.max, e)
                    break
                backoff = min(60, 2 ** min(self.restarts, 6)) + random.uniform(0, 1.5)
                log.error("💥 %s crashed: %s → restart #%d in %.1fs",
                          self.name, e, self.restarts, backoff)
                await asyncio.sleep(backoff)
                set_task_state(self.name, running=True)

    async def stop(self):
        self.stop_evt.set()
        if self.task and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(Exception):
                await self.task

# ───────────── Optional imports (never hard-fail) ────────────
def optional(modpath: str):
    try:
        __import__(modpath)
        return sys.modules[modpath]
    except Exception as e:
        log.info("Optional module unavailable: %s (%s)", modpath, e)
        return None

def discover_symbols(cli_symbols: List[str] | None) -> List[str]:
    if cli_symbols: return cli_symbols
    return DEFAULT_SYMBOLS

# Child task factories (safe, best-effort)
async def task_pulse(symbols: List[str]):
    # Try a broker account probe if user’s module exists
    broker = optional("lavish_core.trade.broker_alpaca")
    while True:
        try:
            equity = buying_power = "-"
            if broker and hasattr(broker, "get_account"):
                acct = await asyncio.to_thread(broker.get_account)
                equity = acct.get("equity", "-")
                buying_power = acct.get("buying_power", "-")
            log.info("💓 pulse | mode=%s | eq=%s | bp=%s | symbols=%s",
                     MODE, equity, buying_power, ",".join(symbols))
            set_task_state("pulse", equity=equity, buying_power=buying_power)
        except Exception as e:
            log.warning("pulse error: %s", e)
            set_task_state("pulse", last_error=str(e))
        await asyncio.sleep(20)

async def task_brain_watch():
    # Prefer the single-file brain_master if present
    # Falls back to superbot/news OCR if needed
    # We simply call its watch/loop entrypoints in threads
    # 1) lavish brain_master
    mod = optional("brain_master")
    if mod and hasattr(mod, "watch_forever"):
        await asyncio.to_thread(mod.watch_forever, REFRESH_MIN)
        return
    # 2) lavish_core.superbot loop (RSS/OCR/memory) as periodic
    superbot_mod = optional("lavish_core.superbot")
    if superbot_mod and hasattr(superbot_mod, "SuperBot"):
        bot = superbot_mod.SuperBot(mode="dry")
        while True:
            try:
                bot.step_collect_news(symbols=None)
                bot.step_ingest_ocr()
                set_task_state("brain_watch", last_run=datetime.now(timezone.utc).isoformat())
            except Exception as e:
                log.warning("brain_watch error: %s", e)
                set_task_state("brain_watch", last_error=str(e))
            await asyncio.sleep(REFRESH_MIN * 60)
    else:
        # Nothing available; idle with warning
        while True:
            log.info("brain_watch idle (no modules found)")
            await asyncio.sleep(60)

async def task_super_pipeline_1(symbols: List[str]):
    mod = optional("lavish_core.super_pipeline")
    if not mod:
        log.info("super_pipeline not found; task will idle.")
        while True:
            await asyncio.sleep(60)
    runner = getattr(mod, "run_pipeline", None) or getattr(mod, "main", None)
    if runner is None:
        log.info("super_pipeline has no run entrypoint; task will idle.")
        while True:
            await asyncio.sleep(60)

    async def _run():
        if getattr(mod, "run_pipeline", None):
            await asyncio.to_thread(mod.run_pipeline, since_hours=24, symbols=symbols, max_items=80)
        else:
            await asyncio.to_thread(mod.main)

    while True:
        try:
            await _run()
            set_task_state("super_pipeline_1", last_run=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            log.error("super_pipeline_1 error: %s", e)
            set_task_state("super_pipeline_1", last_error=str(e))
        # run once per cycle (tune here if needed)
        await asyncio.sleep(REFRESH_MIN * 60)

async def task_vision_batch():
    # Prefer run_vision_batch.py single-file if present
    mod = optional("run_vision_batch")
    if not mod:
        log.info("vision batch not found; task will idle.")
        while True:
            await asyncio.sleep(60)
    # Default: loop mode (uses env OPENAI_API_KEY if present)
    import argparse
    class Args: pass
    while True:
        try:
            # emulate: python run_vision_batch.py --once (quick) then sleep
            if hasattr(mod, "main"):
                os.environ.setdefault("OPENAI_VISION_MODEL", "gpt-4o-mini")
                # one shot batch
                import importlib
                importlib.reload(mod)
                mod.main.__wrapped__ = getattr(mod, "main", None)  # hint for linters
                # Invoke once via args
                sys_argv_backup = sys.argv[:]
                sys.argv = ["run_vision_batch.py", "--once"]
                try:
                    mod.main()
                finally:
                    sys.argv = sys_argv_backup
                set_task_state("vision_batch", last_run=datetime.now(timezone.utc).isoformat())
            else:
                log.info("vision batch module has no main(); idle one minute.")
        except Exception as e:
            log.error("vision_batch error: %s", e)
            set_task_state("vision_batch", last_error=str(e))
        await asyncio.sleep(REFRESH_MIN * 60)

# ───────────────────────── CLI parsing ───────────────────────
import argparse
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lavish run_all (LAN health, dual logging, supervised)")
    p.add_argument("--mode", choices=["dev","paper","live"], default=MODE)
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--enable-vision", action="store_true", default=ENABLE_VISION)
    p.add_argument("--watch", action="store_true", default=ENABLE_WATCH, help="Enable continuous watch mode")
    p.add_argument("--pulse-interval", type=int, default=20)
    p.add_argument("--no-pulse", action="store_true")
    p.add_argument("--host", type=str, default=HEALTH_HOST)
    p.add_argument("--port", type=int, default=HEALTH_PORT)
    return p.parse_args()

# ─────────────────────── Orchestration ───────────────────────
async def start_health_server(host: str, port: int):
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    await server.serve()

async def orchestrate(args):
    # Symbols
    raw_syms = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    symbols = discover_symbols(raw_syms)

    # Banner
    banner = textwrap.dedent(f"""
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Lavish — run_all (Wall Street Edition, LAN Health)   ┃
    ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
    ┃ Mode: {args.mode.upper():<5}   Alpaca: {'paper' if 'paper' in ALPACA_BASE_URL else 'live':<5}
    ┃ Symbols: {', '.join(symbols) if symbols else '[none]'}
    ┃ Vision:  {str(args.enable_vision)}
    ┃ Health:  http://{args.host}:{args.port}/status
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """).strip()
    print(banner)

    # Health server
    health_task = asyncio.create_task(start_health_server(args.host, args.port))

    # Supervised tasks
    tasks: List[Supervisor] = []
    if not args.no_pulse:
        tasks.append(Supervisor("pulse", lambda: task_pulse(symbols)))
    # Brain/news/DB
    tasks.append(Supervisor("brain_watch", task_brain_watch))
    # Pipeline 1 if symbols
    if symbols:
        tasks.append(Supervisor("super_pipeline_1", lambda: task_super_pipeline_1(symbols)))
    # Vision optional
    if args.enable_vision:
        tasks.append(Supervisor("vision_batch", task_vision_batch))

    runners = [asyncio.create_task(t.start()) for t in tasks]

    # Graceful shutdown
    stop_evt = asyncio.Event()
    def _sig(*_):
        log.warning("signal received; stopping…")
        stop_evt.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(s, _sig)

    await stop_evt.wait()

    for t in tasks:
        await t.stop()
    for r in runners:
        with contextlib.suppress(Exception):
            await r
    with contextlib.suppress(Exception):
        health_task.cancel()
        await health_task
    log.info("shutdown complete")

# ────────────────────── Entrypoint ───────────────────────────
if __name__ == "__main__":
    args = parse_args()
    # Log vetted keys (masked), helpful for “did it load my .env?”
    masked = {k: v for k, v in env_snapshot(mask=True).items()
              if any(h in k.upper() for h in SECRET_HINTS)}
    log.info("ENV keys (masked): %s", masked)

    try:
        asyncio.run(orchestrate(args))
    except KeyboardInterrupt:
        pass