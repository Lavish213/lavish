# archive/

Dead code, not the live bot. Confirmed via repo audit: nothing in
`run_ingest.py`'s import chain (the real pipeline - see `system_boot.py`
for the authoritative module list) touches any of this, and nothing else
in the repo imports these files either (`git grep` came back empty for
each before moving). Kept rather than deleted in case something in here
is still wanted later; moved out of the repo root so browsing the project
doesn't read as "this is all one system."

- `main.py`, `main_launcher.py`, `lavish_bootstrap.py` - early/alternate
  entry-point attempts, superseded by `run_ingest.py`.
- `ui.py`, `server.py`, `lavish_gateway.py`, `auto_sync.py` - a separate,
  unfinished web dashboard product (FastAPI/SQLAlchemy/Postgres backend +
  the `lavish_ui/` frontend), unrelated to the trading pipeline. Its
  backend is `lavish_core/legacy_dashboard_api/` (renamed from
  `hybrid_store/` during this audit - it collided in name with the real
  `lavish_core/db/hybrid_store.py`, which the live bot actually uses).
- `brain_master.py`, `yt_brain_master.py`, `lavish_super_vector.py` -
  standalone exploratory scripts (knowledge/news enrichment, YouTube
  intelligence, a parallel SQLite+Alpaca ingestor). Each a self-contained
  alternate take on ideas the real pipeline already covers.
- `super_injector.py` - a one-shot bootstrapper for cloning/updating other
  GitHub repos. Not part of any running system.
- `bot_bridge.py` - a thin optional-dependency wrapper with no real
  pipeline caller.
- `news_crypto.py` - already archived before this audit; left as-is.

Not moved, but also confirmed dead and worth cleaning up later if you
want to go further: top-level `core/` and `agents/` (zero references
anywhere), `lavish_core/trade/decider.py` and `decision_engine.py` (an
ML-based trade-decision gate that was built but never wired into
`trade_handler.py` - confirmed by checking its imports directly),
`lavish_core/trade/alpaca_handler.py` and
`lavish_core/trading/auto_signal_runner.py` (already flagged as orphaned
by comments in `system_boot.py`/`start_bot.sh` before this audit), and
the whole `lavish_core/legacy_dashboard_api/` + `lavish_ui/` pair. These
were left in place rather than moved because full packages have more
surface area for a subtle break (e.g. a relative path or sibling import
this audit didn't trace) than the standalone scripts above did.
