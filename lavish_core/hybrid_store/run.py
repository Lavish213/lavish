import uvicorn

if __name__ == "__main__":
    uvicorn.run("lavish_core.hybrid_store.backend.main:app", host="0.0.0.0", port=8000, reload=True)