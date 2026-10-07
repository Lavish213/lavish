import uvicorn

if __name__ == "__main__":
    uvicorn.run("lavish_core.legacy_dashboard_api.backend.main:app", host="0.0.0.0", port=8000, reload=True)