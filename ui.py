from __future__ import annotations

"""
Lavish UI Gateway – Wall Street Engine

Single entrypoint for:
- Lavish Core API  → /api
- Lavish UI (intro/dashboard/portfolio) → HTML + static assets
"""

import asyncio
import logging
import os
import socket
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ─────────────────────────────────────────────
# 1. ENV + LOGGING
# ─────────────────────────────────────────────

ENV = os.getenv("ENV", "dev").lower()
TZ = os.getenv("TIMEZONE", "America/Los_Angeles")
DEFAULT_PORT = int(os.getenv("PORT", "8000"))

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "lavish_ui_gateway.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | lavish.ui | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("lavish.ui")
log.info("🟣 Booting Lavish UI Gateway…")

# ─────────────────────────────────────────────
# 2. IMPORT CORE API + DB INIT
# ─────────────────────────────────────────────

try:
    from lavish_core.api.api_handler import app as core_app
    log.info("✅ Imported Lavish Core API (lavish_core.api.api_handler)")
except Exception:
    log.exception("❌ Failed to import Lavish Core API")
    raise

try:
    from lavish_core.hybrid_store.backend.database import init_db
    log.info("✅ Imported init_db() from lavish_core.hybrid_store.backend.database")
except Exception:
    log.exception("❌ Failed to import init_db()")
    raise

# ─────────────────────────────────────────────
# 3. UI FOLDERS (REAL lavish_ui, NOT dist)
# ─────────────────────────────────────────────

UI_BASE = BASE_DIR / "lavish_ui"
TEMPLATE_DIR = UI_BASE / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

UI_FOLDERS = {
    "intro": UI_BASE / "intro",
    "shared": UI_BASE / "shared",
    "dashboard": UI_BASE / "dashboard",
    "portfolio": UI_BASE / "portfolio",
    "assets": UI_BASE / "assets",
}

# ─────────────────────────────────────────────
# 4. MAIN FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Lavish UI Gateway",
    version="1.0.0",
    description="Gateway + UI shell for the Lavish Wall Street Engine.",
)

allowed_origins = [
    "*",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the core API at /api
app.mount("/api", core_app)

# ─────────────────────────────────────────────
# 5. STATIC / UI MOUNTS (for CSS/JS/images)
# ─────────────────────────────────────────────

# Individual feature folders (optional but nice to have)
for route_name, folder_path in UI_FOLDERS.items():
    if folder_path.exists():
        # /intro, /dashboard, /portfolio, /shared, /assets
        app.mount(f"/{route_name}", StaticFiles(directory=str(folder_path)), name=route_name)
        log.info(f"✅ Mounted /{route_name} → {folder_path}")
    else:
        log.warning(f"⚠️ Missing UI folder: {folder_path}")

# CRITICAL: /static for Jinja2 → url_for('static', path='...')
assets_path = UI_FOLDERS["assets"]
if assets_path.exists():
    app.mount("/static", StaticFiles(directory=str(assets_path)), name="static")
    log.info(f"✅ Mounted /static → {assets_path}")
else:
    log.warning("⚠️ Assets folder missing – /static URLs may fail")

# ─────────────────────────────────────────────
# 6. HEARTBEAT LOOP
# ─────────────────────────────────────────────

async def heartbeat_loop():
    while True:
        log.info(f"💜 Lavish UI Gateway heartbeat (env={ENV}, tz={TZ})")
        await asyncio.sleep(60)

# ─────────────────────────────────────────────
# 7. STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    log.info("🔹 Initializing Lavish database…")
    try:
        init_db(verify=True)
        log.info("✅ Database initialization OK")
    except Exception as e:
        log.exception(f"❌ DB init failed: {e}")

    asyncio.create_task(heartbeat_loop())
    log.info("🚀 Lavish UI Gateway startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    log.info("🛑 Lavish UI Gateway shutting down")

# ─────────────────────────────────────────────
# 8. ROOT → INTRO SCREEN (intro.html)
# ─────────────────────────────────────────────

@app.get("/")
async def root(request: Request):
    """
    Serve the Lavish intro UI (templates/intro.html).
    This is the 'gold / lavish' landing page.
    """
    intro_template = TEMPLATE_DIR / "intro.html"
    if intro_template.exists():
        return templates.TemplateResponse("intro.html", {"request": request})

    # Fallback if template missing
    return JSONResponse(
        {
            "message": "Lavish UI Gateway is running — intro.html not found.",
            "hint": f"Expected at: {intro_template}",
            "api_root": "/api",
        }
    )

# Optional: explicit route alias for /intro
@app.get("/intro")
async def intro_page(request: Request):
    intro_template = TEMPLATE_DIR / "intro.html"
    if intro_template.exists():
        return templates.TemplateResponse("intro.html", {"request": request})
    return JSONResponse({"detail": "intro.html not found."}, status_code=404)

# ─────────────────────────────────────────────
# 9. HEALTH + STATUS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "env": ENV, "tz": TZ}

@app.get("/status")
async def status():
    return {
        "service": "lavish-ui-gateway",
        "env": ENV,
        "time": datetime.utcnow().isoformat(),
        "ui_paths": {k: str(v) for k, v in UI_FOLDERS.items()},
        "api_root": "/api",
    }

# ─────────────────────────────────────────────
# 10. PORT FINDER + MAIN ENTRYPOINT
# ─────────────────────────────────────────────

def find_open_port(start: int = DEFAULT_PORT, max_tries: int = 20) -> int:
    port = start
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("No free ports available")

if __name__ == "__main__":
    import uvicorn

    port = find_open_port(DEFAULT_PORT)
    log.info(f"🚀 Launching Lavish UI Gateway on port {port}")

    uvicorn.run(
        "ui:app",          # << because this file is ui.py
        host="0.0.0.0",
        port=port,
        reload=(ENV == "dev"),
        log_level="info",
    )