# lavish_core/hybrid_store/backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from the same backend package
from . import health, config, orders, portfolio, signals

# Import DB initializer
from .database import init_db


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid Store API",
        version="1.0.0"
    )

    # CORS Setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize the database schema
    init_db()

    # Register routers
    app.include_router(health.router)
    app.include_router(config.router)
    app.include_router(orders.router)
    app.include_router(portfolio.router)
    app.include_router(signals.router)

    @app.get("/")
    def root():
        return {
            "ok": True,
            "service": "hybrid-store",
            "docs": "/docs"
        }

    return app


# Instantiate app for uvicorn
app = create_app()