import lavish_bootstrap
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print(">>> sys.path includes:", sys.path)

from lavish_core.hybrid_store.backend.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("lavish_core.hybrid_store.backend.main:app", host="127.0.0.1", port=8000, reload=True)
