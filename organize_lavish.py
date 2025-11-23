import os
import shutil

base = os.getcwd()
ui_path = os.path.join(base, "lavish_ui")
templates_path = os.path.join(ui_path, "templates")
static_path = os.path.join(ui_path, "static")
css_path = os.path.join(static_path, "css")
js_path = os.path.join(static_path, "js")

# Create directories if missing
os.makedirs(css_path, exist_ok=True)
os.makedirs(js_path, exist_ok=True)
os.makedirs(templates_path, exist_ok=True)

# Move HTML files to templates
for file in ["index.html", "dashboard.html", "intro.html"]:
    src = os.path.join(base, file)
    if os.path.exists(src):
        shutil.move(src, templates_path)
        print(f"✅ Moved {file} → templates/")

# Move CSS files to static/css
for file in ["style.css", "dashboard.css"]:
    src = os.path.join(base, file)
    if os.path.exists(src):
        shutil.move(src, css_path)
        print(f"✅ Moved {file} → static/css/")

# Move JS files to static/js
for file in ["script.js"]:
    src = os.path.join(base, file)
    if os.path.exists(src):
        shutil.move(src, js_path)
        print(f"✅ Moved {file} → static/js/")

print("🎯 All files organized for Flask/FastAPI layout!")