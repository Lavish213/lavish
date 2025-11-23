"""
Lavish UI — Intro Launcher
--------------------------
Auto-mounts static routes, serves templates, checks structure, and provides diagnostics.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
import os

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR  # lavish_ui/
app = FastAPI(title="Lavish UI", version="1.1")

# -------------------------------
# Template Setup
# -------------------------------
templates_path = ROOT / "templates"
templates = Jinja2Templates(directory=str(templates_path)) if templates_path.exists() else None

# -------------------------------
# Static Route Mapping
# -------------------------------
static_map = {
    "/intro": ROOT / "intro",
    "/shared": ROOT / "shared",
    "/dashboard": ROOT / "dashboard",
    "/portfolio": ROOT / "portfolio",
    "/assets": ROOT / "assets",
}

mounted_routes = {}
missing_routes = {}

print("\n🔍 Lavish UI Startup Check\n" + "-" * 35)
for route, path in static_map.items():
    if path.exists():
        app.mount(route, StaticFiles(directory=str(path)), name=route.strip("/"))
        mounted_routes[route] = str(path)
        print(f"✅ Mounted {route:<12} → {path}")
    else:
        missing_routes[route] = str(path)
        print(f"⚠️  Missing {path}")

# -------------------------------
# Favicon Handler
# -------------------------------
favicon_path = ROOT / "assets" / "favicon.ico"

@app.get("/favicon.ico")
async def favicon():
    if favicon_path.exists():
        return FileResponse(favicon_path)
    return HTMLResponse("<h1>⚠️ Missing favicon.ico</h1>", status_code=404)

# -------------------------------
# UI Pages
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def intro_page(request: Request):
    """Serves intro.html from templates/."""
    if not templates or not (templates_path / "intro.html").exists():
        return HTMLResponse("<h1>⚠️ Missing intro.html in templates/</h1>", status_code=404)
    return templates.TemplateResponse("intro.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    dash_html = ROOT / "dashboard" / "summary" / "summary.html"
    if dash_html.exists():
        return HTMLResponse(dash_html.read_text())
    return HTMLResponse("<h1>⚠️ Missing dashboard/summary/summary.html</h1>", status_code=404)

@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    port_html = ROOT / "portfolio" / "positions" / "positions.html"
    if port_html.exists():
        return HTMLResponse(port_html.read_text())
    return HTMLResponse("<h1>⚠️ Missing portfolio/positions/positions.html</h1>", status_code=404)

# -------------------------------
# Status Endpoint
# -------------------------------
@app.get("/status", response_class=JSONResponse)
async def status_check():
    """Returns live status of mounted static directories."""
    return {
        "app": "Lavish UI",
        "version": "1.1",
        "mounted_routes": mounted_routes,
        "missing_routes": missing_routes,
        "favicon": "✅" if favicon_path.exists() else "⚠️ Missing",
        "templates": "✅" if templates_path.exists() else "⚠️ Missing",
    }

# -------------------------------
# 404 Handler
# -------------------------------
@app.exception_handler(404)
async def not_found(request: Request, exc):
    file_path = templates_path / "404.html" if templates_path else None
    if file_path and file_path.exists():
        return templates.TemplateResponse("404.html", {"request": request}, status_code=404)
    return HTMLResponse("<h1>404 - Page Not Found</h1>", status_code=404)

# -------------------------------
# Launch
# -------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Launching Lavish UI at http://127.0.0.1:{port}\n")
    uvicorn.run("intro:app", host="0.0.0.0", port=port, reload=True)