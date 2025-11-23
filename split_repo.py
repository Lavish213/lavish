import os
import zipfile
from pathlib import Path

ROOT = Path(".")
TARGET = Path("lavish_split")
TARGET.mkdir(exist_ok=True)

# --------- ignore patterns ----------
IGNORE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
IGNORE_FOLDERS = {
    "images_in",
    "images_out",
    "inbox",
    "processed",
    "raw",
    "__pycache__"
}

# -------- zip repo with ZIP64 enabled ----------
def zip_repo():
    zip_path = "lavish_repo.zip"
    print("📦 Zipping repo (IGNORING IMAGES)...")

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True
    ) as z:
        for folder, _, files in os.walk(ROOT):
            folder_name = os.path.basename(folder)

            if folder_name in IGNORE_FOLDERS:
                continue

            for f in files:
                ext = Path(f).suffix.lower()
                if ext in IGNORE_EXT:
                    continue

                full = os.path.join(folder, f)
                arc = full[len(str(ROOT)) + 1:]
                z.write(full, arc)
                print(" +", arc)

    print("✔ ZIP complete:", zip_path)
    return zip_path

# ---------- split ZIP into 100MB parts ----------
def split_zip(zip_file):
    print("✂ Splitting zip...")

    part_size = 100 * 1024 * 1024  # 100 MB

    with open(zip_file, "rb") as f:
        index = 0
        while True:
            chunk = f.read(part_size)
            if not chunk:
                break

            part_name = TARGET / f"lavish_part_{index:02d}.zip"
            with open(part_name, "wb") as p:
                p.write(chunk)

            print(f"   ✔ Made: {part_name}")
            index += 1

    print("\n🎉 Done — upload the files from lavish_split/")

if __name__ == "__main__":
    zip_file = zip_repo()
    split_zip(zip_file)