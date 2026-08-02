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
import threading

from lavish_core.logger_setup import get_logger
from lavish_core.patreon import patreon_trigger
from lavish_core.discord_ingest import listener as discord_listener

log = get_logger("run_ingest", log_dir="logs")


def main() -> None:
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
