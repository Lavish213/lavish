# lavish_bot/ai/watch_folder.py
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# import from the new unified Vision module
from lavish_core.vision.vision_reader import VisionReader

# define the image inbox
WATCH_DIR = Path(__file__).resolve().parents[1] / "inbox"
WATCH_DIR.mkdir(parents=True, exist_ok=True)

# initialize OCR + trade engine
vision = VisionReader()

class AIWatchHandler(FileSystemEventHandler):
    def on_created(self, event):
        p = Path(event.src_path)
        if p.is_dir():
            return
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
            return
        time.sleep(0.25)  # ensure file write completes
        try:
            print(f"[📸 New Image Detected] {p.name}")
            result = vision.process_image(p)
            print(f"[✅ Image Processed] {p.name} → {result}")
        except Exception as e:
            print(f"[❌ Error Processing {p.name}] {e}")

def start_ai_watch():
    """Start the AI watch service to monitor the inbox folder."""
    observer = Observer()
    handler = AIWatchHandler()
    observer.schedule(handler, str(WATCH_DIR), recursive=False)
    observer.start()
    print(f"[👁️  AI Watch Active] Drop stock screenshots into: {WATCH_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_ai_watch()