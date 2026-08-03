# run_ingest.py
# Single entry point for watching both alert sources (Patreon + Discord) at
# once. This has to be one process, not two: they share one DuckDB file
# (lavish_core/db/hybrid_store.py) for the order/fill/signal audit trail and
# the cross-source dedup check in trade_handler.execute_trade_from_post(),
# and DuckDB does not allow two separate processes to hold the same
# database file open concurrently - running the Patreon poller and the
# Discord listener as two "python -m ..." processes would fight over the
# file lock. Patreon polling runs in a background thread; the Discord
# client owns the main thread (discord.py-self manages its own event loop
# and blocks on client.run()).
#
# Discord only actually connects if DISCORD_USER_TOKEN and
# DISCORD_INGEST_ACCEPT_TOS_RISK=1 are both set - see
# lavish_core/discord_ingest/listener.py for why. Without them, this runs
# Patreon-only, same as always.
from __future__ import annotations
import os, threading

from lavish_core.logger_setup import get_logger
from lavish_core.patreon import patreon_trigger
from lavish_core.discord_ingest import listener as discord_listener
from lavish_core.trade.reconcile import reconcile_open_positions, run_periodic_reconciliation
from lavish_core.trade.account_monitor import run_periodic_snapshot

log = get_logger("run_ingest", log_dir="logs")


def _log_startup_config() -> None:
    """
    One consolidated line at boot with the config that actually matters for
    trading behavior - previously the only way to know what a running
    process was actually configured to do was to go read its env vars
    directly; this makes it visible in the log stream itself, which is
    what anyone monitoring the bot is actually tailing.
    """
    whitelist = os.getenv("WHITELIST_TICKERS", "")
    discord_on = bool(discord_listener.DISCORD_USER_TOKEN and discord_listener.ACCEPT_TOS_RISK)
    log.info(
        "=== Lavish_bot startup config === "
        "TRADE_MODE=%s ALPACA_BASE_URL=%s tickers=%d "
        "discord_ingest=%s options_own_stop=%s%% daily_loss_limit=%s%% "
        "weekly_loss_limit=%s%% max_consecutive_losses=%s "
        "max_correlated_positions=%s position_size_scale=%s",
        os.getenv("TRADE_MODE", "dry"),
        os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        len([t for t in whitelist.split(",") if t.strip()]),
        discord_on,
        round(float(os.getenv("OPTIONS_OWN_STOP_LOSS_PCT", "0.50")) * 100, 2),
        round(float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.03")) * 100, 2),
        round(float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", "0.07")) * 100, 2),
        os.getenv("MAX_CONSECUTIVE_LOSSES", "5"),
        os.getenv("MAX_CORRELATED_POSITIONS", "3"),
        os.getenv("POSITION_SIZE_SCALE", "1.0"),
    )
    if os.getenv("TRADE_MODE", "dry").lower() == "live":
        log.warning("TRADE_MODE=live - this process will submit REAL orders with real money.")


def main() -> None:
    _log_startup_config()

    # Every exit-guardrail thread lives only in this process's memory - on
    # every start (including a crash-restart) any option position already
    # open at the broker has no one watching it until this runs. Do it
    # once immediately, then keep checking periodically in case a single
    # monitor thread dies without taking the whole process down with it.
    try:
        recovered = reconcile_open_positions()
        if recovered:
            log.warning("Startup reconciliation recovered %d unmanaged position(s).", recovered)
        else:
            log.info("Startup reconciliation: no unmanaged positions found.")
    except Exception as e:
        log.error("Startup reconciliation failed: %s", e)
    threading.Thread(target=run_periodic_reconciliation, name="reconcile-loop", daemon=True).start()
    threading.Thread(target=run_periodic_snapshot, name="account-snapshot-loop", daemon=True).start()

    patreon_thread = threading.Thread(target=patreon_trigger.poll_loop, name="patreon-poller", daemon=True)
    patreon_thread.start()
    log.info("Patreon poller started in background thread.")

    if discord_listener.DISCORD_USER_TOKEN and discord_listener.ACCEPT_TOS_RISK:
        log.info("Starting Discord listener on the main thread (blocking)...")
        discord_listener.run()
        # client.run() only returns on disconnect/error - if Patreon is
        # still the only thing running at that point, keep the process
        # alive on it rather than exiting.
        log.warning("Discord listener stopped. Falling back to Patreon-only for the rest of this run.")
        patreon_thread.join()
    else:
        log.info("Discord not configured (DISCORD_USER_TOKEN/DISCORD_INGEST_ACCEPT_TOS_RISK unset) - Patreon-only.")
        patreon_thread.join()


if __name__ == "__main__":
    main()
