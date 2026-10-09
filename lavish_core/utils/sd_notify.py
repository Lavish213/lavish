# lavish_core/utils/sd_notify.py
# Minimal systemd readiness/watchdog notification (sd_notify protocol),
# stdlib-only - no new dependency for this. Safe to call unconditionally
# from anywhere, including outside systemd entirely (simulate_alerts.py,
# a bare `python3 run_ingest.py`, tests): NOTIFY_SOCKET is only set when
# systemd actually launched the process under Type=notify, so this is a
# silent no-op in every other context.
#
# Protocol: https://www.freedesktop.org/software/systemd/man/sd_notify.html
# (a single datagram write to a Unix socket named in $NOTIFY_SOCKET - no
# library needed, just a few lines of stdlib socket code).
from __future__ import annotations
import logging
import os
import socket

log = logging.getLogger("sd_notify")


def notify(state: str) -> bool:
    """Returns True if a notification was actually sent (running under
    systemd with NOTIFY_SOCKET set), False if this was a no-op."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]  # abstract namespace socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.connect(addr)
        sock.sendall(state.encode("utf-8"))
        return True
    except Exception as e:
        log.warning("sd_notify(%r) failed: %s", state, e)
        return False
    finally:
        sock.close()


def notify_ready() -> None:
    notify("READY=1")


def notify_watchdog() -> None:
    notify("WATCHDOG=1")
