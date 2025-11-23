import os
from collections import defaultdict

def scan_py_files(base_dir="."):
    file_map = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                file_map[file].append(rel_path)

    print("📂 Python File Map:")
    print("-" * 60)
    for file, paths in sorted(file_map.items()):
        print(f"\n{file}:")
        for p in paths:
            print(f"  └── {p}")

    print("\n🔎 Duplicate Files Found:")
    print("-" * 60)
    for file, paths in file_map.items():
        if len(paths) > 1:
            print(f"{file} → {len(paths)} copies:")
            for p in paths:
                print(f"   - {p}")

if __name__ == "__main__":
    print("🔍 Scanning Python files in current directory...\n")
    scan_py_files("Lavish_bot")
    print("\n✅ Scan complete.")