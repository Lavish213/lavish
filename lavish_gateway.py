from __future__ import annotations

"""
Lavish Gateway – Wall Street Engine (MASTER)

Unified, bulletproof gateway for:
- Lavish Core API      (/api)
- Lavish UI frontend   (all HTML + JS + CSS)
- Health + status
- DB initialization
- Heartbeat logger
- Auto port detection
"""

import asyncio
import logging
import os
import socket
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ─────────────────────────────────────────────
# 1. ENV + LOGGING
# ─────────────────────────────────────────────

ENV = os.getenv("ENV", "dev").lower()
TZ = os.getenv("TIMEZONE", "America/Los_Angeles")
DEFAULT_PORT = int(os.getenv("PORT", "8000"))

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "lavish_master.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | lavish.gateway | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("lavish.gateway")
log.info("🟣 Booting Lavish Gateway…")


# ─────────────────────────────────────────────
# 2. IMPORT CORE API + DB INIT
# ─────────────────────────────────────────────

try:
    from lavish_core.api.api_handler import app as core_api
except Exception:
    log.exception("❌ Failed to import Lavish Core API")
    raise

try:
    from lavish_core.hybrid_store.backend.database import init_db
except Exception:
    log.exception("❌ Failed to import init_db()")
    raise


# ─────────────────────────────────────────────
# 3. UI PATHS / TEMPLATES
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "lavish_ui"
TEMPLATE_DIR = UI_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

STATIC_ROOT = UI_DIR  # serves everything under /static


# ─────────────────────────────────────────────
# 4. MAIN FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Lavish Gateway",
    version="1.0.0",
    description="Gateway + orchestrator for Lavish Wall Street Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach the Lavish Core API
app.mount("/api", core_api)


# ─────────────────────────────────────────────
# 5. STATIC MOUNTS (SAFE + CONSISTENT)
# ─────────────────────────────────────────────

if STATIC_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
    log.info(f"✅ Mounted /static → {STATIC_ROOT}")
else:
    log.warning("⚠️ STATIC_ROOT does not exist")


# ─────────────────────────────────────────────
# 6. UI ROUTER (Bulletproof)
# ─────────────────────────────────────────────

ui = APIRouter()


# / → intro page
@ui.get("/", response_class=HTMLResponse, include_in_schema=False)
async def intro(request: Request):
    intro_file = TEMPLATE_DIR / "intro.html"
    if intro_file.exists():
        return templates.TemplateResponse("intro.html", {"request": request})
    return JSONResponse({"message": "Lavish Gateway running — intro missing"})


# Dashboard page
@ui.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# Home page
@ui.get("/home", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# Achievements page
@ui.get("/achievements", response_class=HTMLResponse, include_in_schema=False)
async def achievements(request: Request):
    return templates.TemplateResponse("achievements.html", {"request": request})


# Global 404 page
@ui.get("/not-found", include_in_schema=False)
async def not_found(request: Request):
    return templates.TemplateResponse("404.html", {"request": request})


app.include_router(ui)


# ─────────────────────────────────────────────
# 7. ICON ROUTES (remove browser noise)
# ─────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon1 = UI_DIR / "icons" / "lavish.ico"
    icon2 = UI_DIR / "assets" / "icons" / "lavish.ico"
    if icon1.exists():
        return FileResponse(icon1)
    if icon2.exists():
        return FileResponse(icon2)
    return JSONResponse(status_code=404, content={"detail": "favicon not found"})


@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_icon():
    icon = UI_DIR / "icons" / "apple-touch-icon.png"
    if icon.exists():
        return FileResponse(icon)
    return JSONResponse(status_code=404, content={"detail": "apple touch icon missing"})


# ─────────────────────────────────────────────
# 8. HEARTBEAT LOOP
# ─────────────────────────────────────────────

async def heartbeat_loop():
    while True:
        log.info(f"💜 Lavish Gateway heartbeat (env={ENV}, tz={TZ})")
        await asyncio.sleep(60)


# ─────────────────────────────────────────────
# 9. STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    log.info("🔹 Initializing DB…")
    try:
        init_db(verify=True)
        log.info("✅ Database OK")
    except Exception:
        log.exception("❌ Database init failed")

    asyncio.create_task(heartbeat_loop())
    log.info("🚀 Gateway startup complete")


@app.on_event("shutdown")
async def shutdown():
    log.info("🛑 Lavish Gateway shutting down")


# ─────────────────────────────────────────────
# 10. HEALTH & STATUS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "env": ENV}


@app.get("/status")
async def status():
    return {
        "service": "lavish-gateway",
        "env": ENV,
        "time": datetime.utcnow().isoformat(),
        "ui_templates": [p.name for p in TEMPLATE_DIR.glob("*.html")],
    }


# ─────────────────────────────────────────────
# 11. PORT FINDER + MAIN ENTRYPOINT
# ─────────────────────────────────────────────

def find_open_port(start: int = DEFAULT_PORT, attempts: int = 20) -> int:
    port = start
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("No free ports found")


if __name__ == "__main__":
    import uvicorn

    port = find_open_port(DEFAULT_PORT)
    log.info(f"🚀 Launching Lavish Gateway on port {port}")

    uvicorn.run(
        "lavish_gateway:app",
        host="0.0.0.0",
        port=port,
        reload=(ENV == "dev"),
        log_level="info",
    )