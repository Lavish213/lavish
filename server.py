from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()

# -----------------------
# STATIC FOLDERS
# -----------------------

# lavish_ui/assets
app.mount(
    "/lavish_ui/assets",
    StaticFiles(directory="lavish_ui/assets"),
    name="lavish_assets"
)

# lavish_ui/shared
app.mount(
    "/lavish_ui/shared",
    StaticFiles(directory="lavish_ui/shared"),
    name="lavish_shared"
)

# lavish_ui/intro
app.mount(
    "/lavish_ui/intro",
    StaticFiles(directory="lavish_ui/intro"),
    name="lavish_intro"
)

# lavish_ui/dashboard
app.mount(
    "/lavish_ui/dashboard",
    StaticFiles(directory="lavish_ui/dashboard"),
    name="lavish_dashboard"
)

# lavish_ui/portfolio
app.mount(
    "/lavish_ui/portfolio",
    StaticFiles(directory="lavish_ui/portfolio"),
    name="lavish_portfolio"
)

# -----------------------
# TEMPLATE FOLDER
# -----------------------
templates = Jinja2Templates(directory="lavish_ui/templates")

# -----------------------
# ROUTES
# -----------------------

# redirect "/" → "/intro"
@app.get("/")
def root():
    return RedirectResponse(url="/intro")


# Intro page
@app.get("/intro")
def intro_page(request: Request):
    return templates.TemplateResponse("intro.html", {"request": request})


# Home page
@app.get("/home")
def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# Dashboard
@app.get("/dashboard")
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# Achievements
@app.get("/achievements")
def achievements_page(request: Request):
    return templates.TemplateResponse("achievements.html", {"request": request})


# 404 page
@app.get("/lost")
def lost(request: Request):
    return templates.TemplateResponse("404.html", {"request": request})


# -----------------------
# RUN SERVER
# -----------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )