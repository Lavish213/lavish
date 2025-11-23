from __future__ import annotations

"""
Lavish Bot – Wall Street API Handler (MAXED EDITION)
----------------------------------------------------

Single entrypoint FastAPI app exposing everything the UI needs:

- Health / config
- Dashboard summary
- Positions, accounts, signals
- Predictions + explanations (“Lavish voice”)
- Trade journal & achievements
- Risk preview / policy checks
- Trainer & sandbox triggers
- Live feed websocket for “heartbeat” panel

This file is **UI-facing only**. Heavy ML, trainers, or brokers are
called via helper modules so they can evolve independently.
"""

import os
import uuid
import time
import socket
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

# ─────────────────────────────────────────────
# Lavish Core imports – adjust paths if needed
# ─────────────────────────────────────────────
from lavish_core.hybrid_store.backend.database import (
    get_session,
    init_db,
)
from lavish_core.hybrid_store.models.position import Position
# from lavish_core.hybrid_store.models.account import Account
# from lavish_core.hybrid_store.models.signal import Signal
# from lavish_core.hybrid_store.models.journal import JournalEntry
# from lavish_core.hybrid_store.models.achievement import Achievement

# from lavish_core.vision.describe import describe_image
# from lavish_core.ai.predictor import predict_outcome


# ─────────────────────────────────────────────
# Settings – read from .env / environment
# ─────────────────────────────────────────────

class LavishSettings(BaseSettings):
    # Core API keys
    openai_api_key: str | None = None
    finnhub_api_key: str | None = None
    finnhub_api_key_2: str | None = None
    alpha_vantage_api_key: str | None = None
    polygon_api_key: str | None = None
    fred_api_key: str | None = None
    newsapi_key: str | None = None
    coingecko_api_key: str | None = None
    coinmarketcap_api_key: str | None = None
    alternative_me_api: str | None = None
    digital_ocean_spaces_key: str | None = None
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None

    # Discord
    discord_channel_id: str | None = None

    # Twilio
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from: str | None = None
    twilio_to: str | None = None

    # Logging
    log_level: str = "INFO"
    debug: bool = False

    # Redis / Database
    redis_url: str = "redis://localhost:6379"
    duckdb_path: str = "./data/storage/lavish.duckdb"
    db_url: str | None = None

    # Postgres
    postgres_host: str | None = None
    postgres_port: str | None = None
    postgres_db: str | None = None
    postgres_user: str | None = None
    postgres_pass: str | None = None

    # Runtime / bot
    bot_refresh_interval_min: int = 60          # replaces BOT_REFRESH_INTERVAL_MIN
    timezone: str = "America/Los_Angeles"       # replaces TIMEZONE
    env: str = "dev"                            # replaces ENV

    class Config:
        env_file = ".env"
        extra = "allow"  # accepts extra keys in .env without errors


settings = LavishSettings()


# ─────────────────────────────────────────────
# Logging – structured & consistent
# ─────────────────────────────────────────────

log = logging.getLogger("lavish.api")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | lavish.api | %(message)s"
        )
    )
    log.addHandler(handler)
    log.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


# ─────────────────────────────────────────────
# FastAPI App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Lavish Bot – Wall Street API",
    version="2.0.0",
    description="Brain + dashboard backend for the Lavish Wall Street UI.",
)

# Allow UI (React, Next.js, etc.) to talk to this API
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins + ["*"],  # loosened; tighten in real prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1024)


# ─────────────────────────────────────────────
# Request ID + latency logging middleware
# ─────────────────────────────────────────────

@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        log.exception(f"[{rid}] Unhandled exception in request pipeline: {exc}")
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"

    log.info(
        f"[{rid}] {request.method} {request.url.path} "
        f"→ {response.status_code} [{duration_ms:.1f}ms]"
    )

    return response


# ─────────────────────────────────────────────
# Global error handler
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "n/a")
    log.exception(f"[{rid}] Global exception at {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Lavish encountered an internal error.",
            "request_id": rid,
        },
    )


# ─────────────────────────────────────────────
# Startup / Shutdown – DB init + internal scheduler
# ─────────────────────────────────────────────

_scheduler_task: Optional[asyncio.Task] = None


async def scheduler_loop():
    """
    Background loop for future:
    - refreshing cached data
    - reconciling logs
    - syncing with news/feeds
    For now just logs a heartbeat.
    """
    interval = max(1, settings.bot_refresh_interval_min) * 60
    while True:
        log.info(
            f"🟣 Lavish Core scheduler tick "
            f"(env={settings.env}, interval={settings.bot_refresh_interval_min}m)"
        )
        await asyncio.sleep(interval)


@app.on_event("startup")
def on_startup() -> None:
    log.info(
        f"🔹 Lavish Core API starting (env={settings.env}, tz={settings.timezone})"
    )
    init_db(verify=True)
    global _scheduler_task
    loop = asyncio.get_event_loop()
    _scheduler_task = loop.create_task(scheduler_loop())
    log.info("✅ Lavish Core API ready.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _scheduler_task
    log.info("🟡 Lavish Core API shutting down…")
    if _scheduler_task:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
    log.info("🔻 Lavish Core API shutdown complete.")


# ─────────────────────────────────────────────
# Pydantic Schemas – Responses & Requests
# ─────────────────────────────────────────────

class HealthStatus(BaseModel):
    status: str = "ok"
    mode: str
    timezone: str
    db: str
    broker_connected: bool = False
    vision_enabled: bool = False


class PositionOut(BaseModel):
    id: int
    symbol: str
    qty: float
    avg_price: float
    market_price: Optional[float] = None
    value: Optional[float] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


class DashboardSummary(BaseModel):
    mode: str
    equity: float
    buying_power: float
    open_positions: int
    day_pnl: float
    watchlist: List[str]
    ai_bias: str
    ai_accuracy_7d: float
    risk_light: str  # "green" | "yellow" | "red"
    sentiment_meter: float  # -1..1
    top_holdings: List[Dict[str, Any]]
    tip_of_the_day: str


class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"
    source: str = Field("core", description="core|stocklady|news|combo")
    image_id: Optional[str] = None
    text_context: Optional[str] = None
    aggression: str = Field("moderate", description="conservative|moderate|aggressive")


class PredictionOut(BaseModel):
    id: str
    symbol: str
    action: str  # buy|sell|hold|avoid
    confidence: float
    grade: str
    reasoning: str
    risk_comment: str
    stop_level: Optional[float]
    target_level: Optional[float]
    source: str
    created_at: datetime


class RiskPreviewRequest(BaseModel):
    symbol: str
    entry: float
    stop: float
    account_equity: float
    max_risk_pct: float = 1.0
    mode: str = "conservative"  # conservative|moderate|aggressive


class RiskPreviewOut(BaseModel):
    symbol: str
    entry: float
    stop: float
    risk_per_share: float
    max_dollar_risk: float
    suggested_size: int
    risk_reward: float
    grade: str
    commentary: str


class JournalEntryOut(BaseModel):
    id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    opened_at: datetime
    closed_at: Optional[datetime]
    source: str
    confidence_at_entry: float
    grade_at_entry: str

    class Config:
        orm_mode = True


class AchievementOut(BaseModel):
    id: int
    code: str
    title: str
    description: str
    unlocked_at: Optional[datetime] = None
    progress: Optional[float] = None

    class Config:
        orm_mode = True


# ─────────────────────────────────────────────
# Helper functions – Lavish Personality, Risk, etc.
# ─────────────────────────────────────────────

def get_mode() -> str:
    # ENV from env var overrides settings.env if present
    return os.getenv("ENV", settings.env).upper()


def risk_light_from_exposure(exposure_pct: float) -> str:
    if exposure_pct < 40:
        return "green"
    if exposure_pct < 70:
        return "yellow"
    return "red"


def synthesize_lavish_commentary(symbol: str, context: Dict[str, Any]) -> str:
    """
    Placeholder for the LLM “Lavish voice”.
    Right now this is simple Python; later you can wire OpenAI/hybrid LLM here.
    """
    direction = context.get("direction", "long").upper()
    confidence = context.get("confidence", 0.5)
    grade = context.get("grade", "B")
    rrr = context.get("rrr", 2.0)

    tone_bits = {
        "A+": "Prime setup. Textbook quality.",
        "A": "Very clean structure with strong confirmation.",
        "B": "Good, but keep risk contained.",
        "C": "Playable, but expect noise.",
        "D": "Edge is thin. Size down hard.",
        "F": "Avoid. This is lottery-ticket territory.",
    }
    tone = tone_bits.get(grade, "Decent, but not perfect.")

    if direction == "LONG":
        dir_line = f"I'm leaning long on {symbol} here."
    elif direction == "SHORT":
        dir_line = f"This looks short-leaning on {symbol}."
    else:
        dir_line = f"No strong directional edge on {symbol} yet."

    return (
        f"{dir_line} Grade: {grade}. Confidence: {round(confidence*100,1)}%. "
        f"Reward-to-risk sits around {rrr:.1f}R. {tone}"
    )


def compute_risk_preview(req: RiskPreviewRequest) -> RiskPreviewOut:
    risk_per_share = abs(req.entry - req.stop)
    if risk_per_share <= 0:
        raise HTTPException(status_code=400, detail="Stop must be away from entry.")

    max_dollar_risk = req.account_equity * (req.max_risk_pct / 100.0)
    suggested_size = int(max_dollar_risk // risk_per_share)

    # very simple R:R stub – real version would look at targets/history
    assumed_target = req.entry + (risk_per_share * 3)
    rrr = (assumed_target - req.entry) / risk_per_share

    # Grade logic – plug in your 100 commandments here later
    if rrr >= 3 and req.max_risk_pct <= 1.0:
        grade = "A"
        commentary = "Risk is tight, reward is 3R+ – this is prime material."
    elif rrr >= 2:
        grade = "B"
        commentary = "Solid structure. Size normal, respect your stop."
    elif rrr >= 1.5:
        grade = "C"
        commentary = "Playable but not special. Consider smaller size."
    else:
        grade = "D"
        commentary = "Edge is thin. Only take this with tiny size or skip."

    return RiskPreviewOut(
        symbol=req.symbol.upper(),
        entry=req.entry,
        stop=req.stop,
        risk_per_share=risk_per_share,
        max_dollar_risk=round(max_dollar_risk, 2),
        suggested_size=max(suggested_size, 0),
        risk_reward=round(rrr, 2),
        grade=grade,
        commentary=commentary,
    )


# ─────────────────────────────────────────────
# Health & Config
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthStatus, tags=["System"])
def health_check(session: Session = Depends(get_session)):
    try:
        session.execute(func.now())
        db_status = "connected"
    except Exception as e:
        log.warning(f"DB health check failed: {e}")
        db_status = "error"

    mode = get_mode()
    tz = settings.timezone

    broker_ok = bool(settings.alpaca_api_key)
    vision_enabled = bool(os.getenv("VISION_ENABLED", "false").lower() == "true")

    return HealthStatus(
        mode=mode,
        timezone=tz,
        db=db_status,
        broker_connected=broker_ok,
        vision_enabled=vision_enabled,
    )


# ─────────────────────────────────────────────
# Root / ping
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "ok": True,
        "service": "lavish-wall-street-api",
        "docs": "/docs",
        "mode": get_mode(),
        "env": settings.env,
        "tz": settings.timezone,
    }


@app.get("/ping", tags=["System"])
async def ping():
    start = time.perf_counter()
    await asyncio.sleep(0)
    duration_ms = (time.perf_counter() - start) * 1000
    return {
        "pong": True,
        "latency_ms": round(duration_ms, 3),
        "env": settings.env,
    }


# ─────────────────────────────────────────────
# Dashboard / Home Overview
# ─────────────────────────────────────────────

@app.get(
    "/dashboard/summary",
    response_model=DashboardSummary,
    tags=["Dashboard"],
)
def get_dashboard_summary(session: Session = Depends(get_session)):
    """
    Aggregate everything the home screen needs in one hit.
    """
    # TODO: replace with real Account model if you have it
    equity = 0.0
    buying_power = 0.0

    positions = session.query(Position).all()
    open_positions = len(positions)

    exposure = sum((p.value or 0.0) for p in positions)
    exposure_pct = 0.0
    if equity > 0:
        exposure_pct = (exposure / equity) * 100

    risk_light = risk_light_from_exposure(exposure_pct)

    # placeholder 7-day accuracy & sentiment meter
    ai_accuracy_7d = 0.72
    sentiment_meter = 0.15  # mildly bullish

    # build top holdings view
    top_holdings = sorted(
        [
            {
                "symbol": p.symbol,
                "value": p.value or 0.0,
                "qty": p.qty,
                "avg_price": p.avg_price,
            }
            for p in positions
        ],
        key=lambda x: x["value"],
        reverse=True,
    )[:3]

    watchlist = [p.symbol for p in positions][:8]

    tip = (
        "Cut losers fast, let winners breathe. Protect capital first, "
        "Lavish will find you another A+ setup."
    )

    return DashboardSummary(
        mode=get_mode(),
        equity=float(round(equity, 2)),
        buying_power=float(round(buying_power, 2)),
        open_positions=open_positions,
        day_pnl=0.0,  # TODO: wire real PnL
        watchlist=watchlist,
        ai_bias="Bullish" if sentiment_meter > 0.1 else "Neutral",
        ai_accuracy_7d=ai_accuracy_7d,
        risk_light=risk_light,
        sentiment_meter=sentiment_meter,
        top_holdings=top_holdings,
        tip_of_the_day=tip,
    )


# ─────────────────────────────────────────────
# Positions & Portfolio
# ─────────────────────────────────────────────

@app.get(
    "/portfolio/positions",
    response_model=List[PositionOut],
    tags=["Portfolio"],
)
def list_positions(session: Session = Depends(get_session)):
    rows = session.query(Position).order_by(Position.symbol.asc()).all()
    return rows


@app.get(
    "/portfolio/positions/{symbol}",
    response_model=PositionOut,
    tags=["Portfolio"],
)
def get_position(symbol: str, session: Session = Depends(get_session)):
    row = (
        session.query(Position)
        .filter(Position.symbol == symbol.upper())
        .one_or_none()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Position not found.")
    return row


# ─────────────────────────────────────────────
# Predictions & AI Voice
# ─────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionOut,
    tags=["AI Predictor"],
)
def make_prediction(req: PredictionRequest, session: Session = Depends(get_session)):
    """
    High-level prediction endpoint.
    All the heavy lifting (vision, NLP, model inference) should happen
    inside your core predictor module. Here we just orchestrate.
    """
    # TODO: call real predictor – this is scaffolding only
    now = datetime.utcnow()
    fake_conf = 0.78
    fake_grade = "A"

    lav_comment = synthesize_lavish_commentary(
        req.symbol,
        {
            "direction": "long",
            "confidence": fake_conf,
            "grade": fake_grade,
            "rrr": 2.8,
        },
    )

    return PredictionOut(
        id=str(uuid.uuid4()),
        symbol=req.symbol.upper(),
        action="buy",
        confidence=fake_conf,
        grade=fake_grade,
        reasoning=lav_comment,
        risk_comment="Risk is acceptable if you keep size within policy and respect the stop.",
        stop_level=None,
        target_level=None,
        source=req.source,
        created_at=now,
    )


# ─────────────────────────────────────────────
# Risk Preview – “What happens if I take this?”
# ─────────────────────────────────────────────

@app.post(
    "/risk/preview",
    response_model=RiskPreviewOut,
    tags=["Risk Engine"],
)
def preview_risk(req: RiskPreviewRequest):
    return compute_risk_preview(req)


# ─────────────────────────────────────────────
# Journal & Achievements – stubs to plug into real models
# ─────────────────────────────────────────────

@app.get(
    "/journal/recent",
    response_model=List[JournalEntryOut],
    tags=["Journal"],
)
def recent_journal(
    limit: int = 20,
    session: Session = Depends(get_session),
):
    """
    Return the most recent trades / decisions.
    Requires a JournalEntry model; replace with your schema.
    """
    # rows = (
    #     session.query(JournalEntry)
    #     .order_by(desc(JournalEntry.opened_at))
    #     .limit(limit)
    #     .all()
    # )
    rows: List[JournalEntryOut] = []  # placeholder
    return rows


@app.get(
    "/achievements",
    response_model=List[AchievementOut],
    tags=["Achievements"],
)
def list_achievements(session: Session = Depends(get_session)):
    # rows = session.query(Achievement).order_by(Achievement.code.asc()).all()
    rows: List[AchievementOut] = []  # placeholder
    return rows


# ─────────────────────────────────────────────
# Trainer / Sandbox Hooks
# ─────────────────────────────────────────────

@app.post("/trainer/refresh", tags=["Trainer"])
def trigger_trainer_refresh():
    """
    Placeholder endpoint to kick off your trainer/sandbox pipeline.
    In production this should enqueue a job (Celery, RQ, etc.) or
    call a separate microservice.
    """
    return {"status": "ok", "message": "Trainer refresh queued."}


@app.post("/sandbox/backtest", tags=["Trainer"])
def run_backtest(payload: Dict[str, Any]):
    """
    Stub for backtesting custom rules or modes.
    The frontend can send strategy parameters and you return a summary.
    """
    return {
        "status": "ok",
        "params": payload,
        "summary": {
            "win_rate": 0.68,
            "avg_rr": 2.4,
            "max_drawdown": -0.12,
            "comment": "Numbers are decent; monitor live before sizing up.",
        },
    }


# ─────────────────────────────────────────────
# Live Feed WebSocket – heartbeat for UI
# ─────────────────────────────────────────────

class LiveClientManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(ws)


live_manager = LiveClientManager()


@app.websocket("/ws/live")
async def live_feed(websocket: WebSocket):
    """
    Live mode websocket.

    Frontend can connect here to show heartbeat events, new
    predictions, alerts, etc. Right now it's a scaffold; later
    you can push real-time events from your core bot.
    """
    await live_manager.connect(websocket)
    try:
        await websocket.send_json(
            {"type": "hello", "message": "Lavish live feed connected."}
        )
        while True:
            # For now, just keep connection open and read pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        live_manager.disconnect(websocket)


# ─────────────────────────────────────────────
# Standalone dev entrypoint (auto-open port)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    preferred_port = int(os.getenv("LAVISH_API_PORT", "8000"))
    port = preferred_port
    log.info(f"🚀 Launching Lavish Core API standalone on :{port}")
    uvicorn.run(
        "lavish_core.api.api_handler:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
    )