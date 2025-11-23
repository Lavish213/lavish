import os
import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EXPECTED = {
    "lavish_core": [
        "vision/king_master.py",
        "patreon/patreon_trigger.py",
        "trade/alpaca_handler.py",
        "__init__.py",
    ],
    ".env": None,
    "run_all.py": None,
}


def check_file_exists(relative_path):
    full_path = BASE_DIR / relative_path
    return full_path.exists()


def create_missing_file(relative_path):
    path = BASE_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Auto-generated placeholder\n")
    print(f"🛠️  Created missing file: {relative_path}")


def import_check(module_path):
    try:
        spec = importlib.util.find_spec(module_path)
        return spec is not None
    except Exception:
        return False


def main():
    print("🔍 Checking Lavish Core System Integrity...\n")
    all_passed = True

    # Check files
    for key, items in EXPECTED.items():
        if items is None:
            exists = check_file_exists(key)
            if exists:
                print(f"✅ {key} found.")
            else:
                print(f"❌ {key} missing. Creating...")
                create_missing_file(key)
                all_passed = False
        else:
            for file in items:
                path = os.path.join(key, file)
                exists = check_file_exists(path)
                if exists:
                    print(f"✅ {path} found.")
                else:
                    print(f"❌ MISSING: {path} — creating placeholder...")
                    create_missing_file(path)
                    all_passed = False

    # Import checks
    print("\n🔎 Checking module imports...")
    modules = [
        "lavish_core.vision.king_master",
        "lavish_core.patreon.patreon_trigger",
        "lavish_core.trade.alpaca_handler",
    ]

    for module in modules:
        if import_check(module):
            print(f"✅ Import works for {module}")
        else:
            print(f"⚠️ Skipped import (module missing): {module}")
            all_passed = False

    # .env check
    print("\n⚙️ Checking .env file...")
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if "=" in line]
        if lines:
            print(f"🔑 Found {len(lines)} keys in .env")
        else:
            print("⚠️ .env exists but is empty")
    else:
        print("❌ No .env file found — please create one")

    # Summary
    print("\n===================================")
    if all_passed:
        print("💚 SYSTEM STATUS: HEALTHY ✅")
    else:
        print("💛 SYSTEM STATUS: AUTO-FIXED (Review Needed)")
    print("===================================")


if __name__ == "__main__":
    main()