# Covers the systemd watchdog integration added for live-money deployment
# hardening. Verified against a real Unix datagram socket (not mocked) -
# this is the actual sd_notify wire protocol, not an assumption about it.
from __future__ import annotations

import os
import socket
import threading

from lavish_core.utils import sd_notify


def test_notify_is_a_safe_noop_without_notify_socket(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert sd_notify.notify("READY=1") is False


def test_notify_sends_the_correct_message_over_a_real_socket(tmp_path, monkeypatch):
    sock_path = str(tmp_path / "notify.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(sock_path)
    try:
        received = []

        def _listen():
            data, _ = server.recvfrom(1024)
            received.append(data.decode())

        t = threading.Thread(target=_listen)
        t.start()

        monkeypatch.setenv("NOTIFY_SOCKET", sock_path)
        result = sd_notify.notify("READY=1")
        t.join(timeout=2)

        assert result is True
        assert received == ["READY=1"]
    finally:
        server.close()


def test_notify_watchdog_sends_the_watchdog_message(tmp_path, monkeypatch):
    sock_path = str(tmp_path / "notify.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(sock_path)
    try:
        received = []

        def _listen():
            data, _ = server.recvfrom(1024)
            received.append(data.decode())

        t = threading.Thread(target=_listen)
        t.start()

        monkeypatch.setenv("NOTIFY_SOCKET", sock_path)
        sd_notify.notify_watchdog()
        t.join(timeout=2)

        assert received == ["WATCHDOG=1"]
    finally:
        server.close()


def test_notify_failure_does_not_raise(monkeypatch):
    # A bad/stale socket path must degrade to "didn't send", never crash
    # the caller - this runs inside the bot's main thread.
    monkeypatch.setenv("NOTIFY_SOCKET", "/nonexistent/path/not/a/real/socket")
    assert sd_notify.notify("READY=1") is False
