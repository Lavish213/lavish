"""
Lavish UI : Structure Verifier
------------------------------
Checks that all key folders and files exist for FastAPI routes.
Prints a color-coded report (✅ found / ⚠️ missing).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

expected_structure = {
    "intro": ["intro.css", "intro.js", "index.html"],
    "shared": ["layout.css", "lighting.css", "animations.css", "lavish_theme.css"],
    "dashboard": ["dashboard.css", "dashboard.js", "summary/summary.html"],
    "portfolio": ["portfolio.js", "positions/positions.html"],
    "assets/fonts": ["Orbitron-Variable.woff2"],
    "templates": ["intro.html", "404.html"]
}

# ----------------------------------------------------------
# Check function
# ----------------------------------------------------------
def check_structure():
    print("\n🔍 Checking Lavish UI Folder Integrity...\n")

    missing_any = False

    for folder, files in expected_structure.items():
        folder_path = BASE_DIR / folder
        if not folder_path.exists():
            print(f"⚠️  Missing folder: {folder_path}")
            missing_any = True
            continue

        print(f"📁 {folder_path.relative_to(BASE_DIR)} exists ✅")

        for f in files:
            file_path = folder_path / f
            if file_path.exists():
                print(f"   ✅ {f}")
            else:
                print(f"   ⚠️  Missing: {f}")
                missing_any = True

    if not missing_any:
        print("\n🎯 All required folders and files found. Lavish UI structure is perfect.\n")
    else:
        print("\n⚠️  Some folders or files are missing. Fix them before running intro.py.\n")

# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
if __name__ == "__main__":
    check_structure()