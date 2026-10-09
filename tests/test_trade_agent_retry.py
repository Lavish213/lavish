# Covers the blind-retry bug: a failed order POST used to be retried
# unconditionally, with nothing to distinguish "the request never reached
# Alpaca" from "it reached Alpaca and only the response was lost" - the
# second case risked a real duplicate order, or (if Alpaca's own
# duplicate-client-id rejection fired) eventually got marked "rejected"
# locally even though a real order existed at the broker. Fixed by
# checking get_order_by_client_id before every retry.
from __future__ import annotations

from unittest.mock import patch

import lavish_core.trade.trade_agent as ta


def _no_sleep(*a, **k):
    pass


def test_retries_and_succeeds_when_truly_never_submitted():
    with patch.object(ta, "time") as mock_time, \
         patch.object(ta, "broker_get_order_by_client_id", return_value=None) as mock_lookup:
        mock_time.sleep.side_effect = _no_sleep
        submit = iter([RuntimeError("timeout"), RuntimeError("timeout"), {"id": "real-order-1", "status": "filled"}])

        def _submit_fn():
            v = next(submit)
            if isinstance(v, Exception):
                raise v
            return v

        result = ta._submit_with_retries(_submit_fn, "client-abc")

    assert result == {"id": "real-order-1", "status": "filled"}
    assert mock_lookup.call_count == 2  # once per failure before the eventual success


def test_finds_existing_order_instead_of_resubmitting():
    # First attempt "fails" (response lost), but the order actually
    # reached Alpaca - lookup must find it and use it, not submit again.
    call_count = {"n": 0}

    def _submit_fn():
        call_count["n"] += 1
        raise RuntimeError("connection reset")

    with patch.object(ta, "time") as mock_time, \
         patch.object(ta, "broker_get_order_by_client_id",
                       return_value={"id": "real-order-1", "status": "filled"}) as mock_lookup:
        mock_time.sleep.side_effect = _no_sleep
        result = ta._submit_with_retries(_submit_fn, "client-abc")

    assert result == {"id": "real-order-1", "status": "filled"}
    assert call_count["n"] == 1  # only the original attempt - no blind resubmit
    mock_lookup.assert_called_once_with("client-abc")


def test_exhausts_retries_and_raises_if_genuinely_unreachable():
    with patch.object(ta, "time") as mock_time, \
         patch.object(ta, "broker_get_order_by_client_id", return_value=None):
        mock_time.sleep.side_effect = _no_sleep

        def _submit_fn():
            raise RuntimeError("Alpaca down")

        try:
            ta._submit_with_retries(_submit_fn, "client-abc")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "Broker retries exhausted" in str(e)
