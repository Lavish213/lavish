from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

# --------------------------------------
# TEMPLATE DIRECTORY DISCOVERY (Correct)
# --------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

ui_router = APIRouter(tags=["UI"])

# --------------------------------------
# INTRO PAGE ROUTE
# --------------------------------------
@ui_router.get("/intro", response_class=HTMLResponse)
async def intro_page(request: Request):
    return templates.TemplateResponse("intro.html", {"request": request})

# --------------------------------------
# HOME PAGE TEST ROUTE (optional)
# --------------------------------------
@ui_router.get("/home", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
