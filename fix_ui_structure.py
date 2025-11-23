import os, shutil

base = os.getcwd()
ui = os.path.join(base, "lavish_ui")
templates = os.path.join(ui, "templates")
static = os.path.join(ui, "static")
css = os.path.join(static, "css")
js = os.path.join(static, "js")

for d in (templates, css, js):
    os.makedirs(d, exist_ok=True)

moves = {
    "index.html": templates,
    "dashboard.html": templates,
    "style.css": css,
    "dashboard.css": css,
    "script.js": js,
    "dashboard.js": js
}

for f, dest in moves.items():
    for root, _, files in os.walk(ui):
        if f in files:
            src = os.path.join(root, f)
            dst = os.path.join(dest, f)
            if src != dst:
                shutil.move(src, dst)
                print(f"✅ moved {f} → {dest}")

print("🎯 lavish_ui organized successfully")