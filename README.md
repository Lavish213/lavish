# Lavish_bot

A bot that watches Patreon/Discord creator thestockalley's trade alerts
(text or trade-card screenshots), parses them (ticker, side, strike,
expiry, target, stop), and auto-executes matching trades on Alpaca
(paper or live) - both equities and options. It copies her stated calls;
it does not generate its own entries or exits from technical analysis.

## Real entry point

`run_ingest.py` - watches Patreon (always) and Discord (optional, see
`lavish_core/discord_ingest/listener.py`'s module docstring first) in one
process, routes alerts through `lavish_core/trading/alert_handler.py` ->
`trade_handler.py` -> `broker_alpaca.py`/`options_broker.py`, with
guardrails (circuit breaker, portfolio correlation limit, PDT guard, our
own options exit monitor) checked before every entry.

- **Deploying for real**: see `DEPLOY.md`.
- **Pre-flight check**: `python3 system_boot.py` - validates credentials,
  the real pipeline's imports, and that the `tesseract` OCR binary is on
  PATH. Must pass before starting the service.
- **Testing without a broker connection**: `python3 simulate_alerts.py`
  (dry run by default - parses sample alerts, no network, no orders).
- **Automated tests**: `pytest tests/`.
- **Config reference**: `env.example` - copy to `.env`, fill in real
  values there, never in chat or in version control.

## Repo layout note

This repo also contains a large amount of orphaned/experimental code
(an unfinished web dashboard, standalone exploratory scripts, an ML
trade-decision gate that was built but never wired in) that predates or
ran parallel to the pipeline above and isn't part of it. Most of it has
been moved to `archive/` (see `archive/README.md` for what's there and
why); anything still flagged dead-but-not-yet-moved is noted there too.
If you're trying to understand what actually runs the bot, start from
`run_ingest.py` and `system_boot.py`'s own import list, not from
browsing the repo top-down.
