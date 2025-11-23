from __future__ import annotations

import os
from flask import Flask, render_template

# Base paths (repo root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "lavish_ui", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "lavish_ui", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR,
)

# ---------------- Routes ---------------- #

@app.route("/")
def intro():
    """
    Lavish animated intro (Matrix + liquid gold LAVISH).
    Uses: lavish_ui/templates/index.html
    """
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """
    Lavish main dashboard shell.
    Uses: lavish_ui/templates/dashboard.html
    """
    return render_template("dashboard.html")


# --------------- Main ---------------- #

if __name__ == "__main__":
    # Dev server only – good for local testing
    app.run(host="0.0.0.0", port=5800, debug=True)