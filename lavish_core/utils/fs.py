from pathlib import Path

def ensure_parent(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)