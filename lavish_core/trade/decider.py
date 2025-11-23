# lavish_core/trade/decider.py
# Lightweight decision/risk gate that sits in front of the main decision_engine + trade_agent

from __future__ import annotations
import os, json, logging
from typing import Dict, Any, Callable, Optional

# ───────────────────────────────────────────────────────────────
# Logging setup
# ───────────────────────────────────────────────────────────────
LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "decider.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("decider")

# ───────────────────────────────────────────────────────────────
# Config knobs (env-tunable)
# ───────────────────────────────────────────────────────────────
MAX_DOLLARS_PER_TRADE = float(os.getenv("MAX_DOLLARS_PER_TRADE", "1500"))
MAX_OPEN_POSITIONS    = int(os.getenv("MAX_OPEN_POSITIONS", "8"))
BLOCKLIST             = {
    s.strip().upper()
    for s in os.getenv("TICKER_BLOCKLIST", "").split(",")
    if s.strip()
}

# ───────────────────────────────────────────────────────────────
# Imports from your ML + decision engine
# ───────────────────────────────────────────────────────────────
try:
    # Buffed predictor we wrote in lavish_core/ml/predictor.py
    from ml.predictor import predict_text
except Exception as e:
    raise ImportError(f"Cannot import predict_text from lavish_core.ml.predictor: {e}")

# decision_engine is optional – we fall back gracefully if missing
try:
    from lavish_core.trade import decision_engine as de  # must have decide_trade()
    HAS_DECISION_ENGINE = True
except Exception:
    HAS_DECISION_ENGINE = False


class TradeDecider:
    """
    Hybrid C-mode:
      • This class is a LIGHT risk/filters gate + glue.
      • It:
          - runs the text predictor on OCR/text
          - applies simple account-level risk checks (blocklist, max $/trade, max open positions)
          - optionally calls decision_engine.decide_trade for deeper logic (trend_z, vol_z, etc.)
          - returns a unified decision dict.

    positions_provider(): -> dict like {"AAPL": {"qty": 10, "avg_cost": 189.2}, ...}
    price_provider(ticker): -> float last/mark price
    equity_provider(): -> float current account equity (optional but recommended)
    """

    def __init__(
        self,
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        price_provider: Optional[Callable[[str], Optional[float]]] = None,
        equity_provider: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        self.positions_provider = positions_provider or (lambda: {})
        self.price_provider     = price_provider     or (lambda t: None)
        self.equity_provider    = equity_provider    or (lambda: None)

    # ───────────────────────────────────────────────────────────
    # High-level, fast risk check (pre-gate before deep logic)
    # ───────────────────────────────────────────────────────────
    def _risk_check(self, symbol: str, action: str) -> Dict[str, Any]:
        """
        Basic sanity + caps:
          - blocklist
          - max open positions (for new BUYs)
          - require a price for BUYs
          - compute max_shares based on MAX_DOLLARS_PER_TRADE
        """
        if not symbol:
            return {"ok": False, "why": "no-symbol"}

        ticker = symbol.upper()

        if ticker in BLOCKLIST:
            return {"ok": False, "why": "blocklist"}

        positions = self.positions_provider() or {}
        open_count = sum(
            1 for _sym, meta in positions.items()
            if abs(float(meta.get("qty", 0))) > 1e-6
        )

        if open_count >= MAX_OPEN_POSITIONS and action.lower() == "buy":
            return {"ok": False, "why": "max-open-positions"}

        price = self.price_provider(ticker) or 0.0
        if price <= 0.0 and action.lower() == "buy":
            return {"ok": False, "why": "no-price"}

        max_shares = None
        if action.lower() == "buy" and price > 0:
            max_shares = int(MAX_DOLLARS_PER_TRADE // max(price, 1e-6))

        return {
            "ok": True,
            "why": "pass",
            "max_shares": max_shares,
            "mark": price,
        }

    # ───────────────────────────────────────────────────────────
    # Core API
    # ───────────────────────────────────────────────────────────
    def decide(self, parsed_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        parsed_payload can contain:
          - 'ticker' (symbol)
          - 'ocr_text' or 'text' (raw text / OCR output)
          - 'math_sig' (optional dict from your quant engine: trend_z, vol_z, etc.)
          - 'equity' (optional; if absent we try equity_provider())
          - optional fields from upstream signals

        Returns a dict:
        {
          'execute': bool,
          'reason': str,
          'symbol': str,
          'side': 'buy'|'sell'|'hold',
          'qty': int,
          'price': float|None,
          'tp': float|None,
          'sl': float|None,
          'model': {... predictor output ...},
          'math_sig': {...} or None,
          'source': 'predictor'|'predictor+math'|'fallback'
        }
        """
        # ── 1) Extract basic fields ─────────────────────────────
        symbol = (parsed_payload.get("ticker") or "").upper()
        text   = parsed_payload.get("ocr_text") or parsed_payload.get("text") or ""
        math_sig = parsed_payload.get("math_sig")  # may be None

        # ── 2) Run text predictor ──────────────────────────────
        model_out = predict_text(text) if text else {"label": "UNKNOWN", "confidence": 0.0, "probs": {}}
        label = model_out.get("label", "UNKNOWN").upper()
        conf  = float(model_out.get("confidence", 0.0))

        # Map label → side
        if label == "BUY":
            side = "buy"
        elif label == "SELL":
            side = "sell"
        else:
            side = "hold"

        # ── 3) Pre-risk filter (fast gate) ─────────────────────
        r = self._risk_check(symbol, side)
        if not r.get("ok"):
            decision = {
                "execute": False,
                "reason": f"risk:{r.get('why')}",
                "symbol": symbol,
                "side": side,
                "qty": 0,
                "price": r.get("mark", None),
                "tp": None,
                "sl": None,
                "model": model_out,
                "math_sig": math_sig,
                "source": "predictor",
            }
            logger.info(json.dumps(decision))
            return decision

        # ── 4) Fetch equity + current position ─────────────────
        equity = parsed_payload.get("equity")
        if equity is None:
            try:
                equity = self.equity_provider() or 0.0
            except Exception:
                equity = 0.0

        positions = self.positions_provider() or {}
        current_pos = positions.get(symbol, None)

        # ── 5) Optional deep decision via decision_engine ──────
        tp = None
        sl = None
        qty = 0

        if HAS_DECISION_ENGINE and equity and equity > 0:
            # We treat the predictor output as a "patreon-like" signal.
            patreon = {
                "symbol": symbol,
                "action": side,          # 'buy'|'sell'|'hold'
                "confidence": conf,      # 0..1
                "price": r.get("mark"),  # mark price if we have it
            }

            try:
                de_decision = de.decide_trade(
                    equity=float(equity),
                    patreon=patreon,
                    math_sig=math_sig,
                    current_position=current_pos,
                )
                side = de_decision.get("side", side)
                qty  = int(de_decision.get("qty", 0) or 0)
                tp   = de_decision.get("tp")
                sl   = de_decision.get("sl")
                # We keep blended_confidence etc. but don't have to expose all here
                source = "predictor+math" if math_sig else "predictor"
            except Exception as e:
                # Fallback to simple mode
                logger.warning(f"decision_engine failed, falling back to simple sizing: {e}")
                source = "fallback"
        else:
            source = "predictor"

        # ── 6) Fallback sizing if decision_engine not used/failed ───
        if qty <= 0:
            if side == "buy":
                max_shares = r.get("max_shares") or 0
                qty = max(1, int(max_shares)) if max_shares > 0 else 0
            elif side == "sell":
                pos = current_pos or {}
                qty = int(abs(float(pos.get("qty", 0))) or 0)

        price = r.get("mark") or None

        # If still no shares or side is hold → execute = False
        execute = bool(side in ("buy", "sell") and qty > 0)

        # ── 7) Final decision packet ───────────────────────────
        decision = {
            "execute": execute,
            "reason": "ok" if execute else "no-qty-or-hold",
            "symbol": symbol,
            "side": side,
            "qty": int(qty or 0),
            "price": price,
            "tp": tp,
            "sl": sl,
            "model": model_out,
            "math_sig": math_sig,
            "source": source,
        }

        logger.info(json.dumps(decision))
        return decision