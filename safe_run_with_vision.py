#!/usr/bin/env python3
"""
Safe launcher that prevents recursive --enable-vision spawns.
Automatically sets env vars and launches Lavish once.
"""

import os, sys, subprocess

def main():
    # --- guard: prevent infinite recursion ---
    if os.getenv("LAVISH_ALREADY_RUNNING") == "1":
        print("🚫 Lavish already running, skipping duplicate launch.")
        return

    os.environ["LAVISH_ALREADY_RUNNING"] = "1"

    # --- set default vision flags ---
    os.environ.setdefault("ENABLE_VISION", "1")
    os.environ.setdefault("FORCE_ENABLE_VISION", "1")

    # --- prepare python exec command ---
    py = sys.executable
    cmd = [py, "-m", "lavish_core.run", "--enable-vision"]

    print("🚀 Launching Lavish safely with vision enabled...")
    subprocess.run(cmd, check=False, env=os.environ)

if __name__ == "__main__":
    main()