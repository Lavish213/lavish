# Covers the two exit-confirmation bugs found and fixed this session:
# (1) a submitted-but-unfilled closing order was marked "exited" with no
#     fill confirmation at all, and (2) a failed order submission ended
#     the monitor thread entirely instead of retrying. Both tested by
#     driving watch_and_exit() directly with mocked broker calls and
#     time.sleep patched to a no-op (polling intervals shouldn't make
#     tests slow).
from __future__ import annotations

from unittest.mock import patch

import lavish_core.trade.options_exit_monitor as oem


def _no_sleep(*a, **k):
    pass


def test_our_stop_loss_triggers_and_confirms_real_fill_price():
    # Entry 10.00, quoted mid 4.00 -> -60%, past the -50% default stop.
    # The order is accepted as pending, THEN confirmed filled on a later
    # poll at a different (real) price - the result must use that real
    # fill price, not the originally quoted limit price.
    with patch.object(oem, "time") as mock_time, \
         patch.object(oem, "latest_quote", return_value=100.0), \
         patch.object(oem, "latest_option_quote", return_value={"bid": 3.9, "ask": 4.1}), \
         patch.object(oem, "place_option_order", return_value={"id": "o1", "status": "pending_new"}) as mock_place, \
         patch.object(oem, "get_order", side_effect=[
             {"status": "pending_new"},
             {"status": "filled", "filled_avg_price": "3.95", "filled_qty": "1"},
         ]) as mock_get, \
         patch.object(oem, "cancel_order") as mock_cancel, \
         patch.object(oem, "post_discord"):
        mock_time.sleep.side_effect = _no_sleep
        mock_time.time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        result = oem.watch_and_exit(
            underlying_symbol="TSLA", contract_symbol="TSLA260101C00100000",
            qty=1, option_side="call", entry_price=10.0,
        )

    assert result["status"] == "exited"
    assert result["reason"] == "our_stop_loss"
    assert result["option_price"] == 3.95  # the CONFIRMED fill price, not the quoted 4.0 limit
    mock_place.assert_called_once()
    assert mock_get.call_count == 2
    mock_cancel.assert_not_called()


def test_exit_retries_after_submission_failure_instead_of_dying():
    with patch.object(oem, "time") as mock_time, \
         patch.object(oem, "latest_quote", return_value=100.0), \
         patch.object(oem, "latest_option_quote", return_value={"bid": 3.9, "ask": 4.1}), \
         patch.object(oem, "place_option_order", side_effect=[
             RuntimeError("Alpaca 503"),
             RuntimeError("Alpaca 503"),
             {"id": "o1", "status": "filled", "filled_avg_price": "4.0", "filled_qty": "1"},
         ]) as mock_place, \
         patch.object(oem, "get_order") as mock_get, \
         patch.object(oem, "post_discord") as mock_discord:
        mock_time.sleep.side_effect = _no_sleep
        mock_time.time.side_effect = list(range(20))

        result = oem.watch_and_exit(
            underlying_symbol="TSLA", contract_symbol="TSLA260101C00100000",
            qty=1, option_side="call", entry_price=10.0,
        )

    # A buggy version would have returned "exit_order_failed" (or crashed
    # the thread) on the first RuntimeError. This must keep retrying and
    # eventually succeed.
    assert result["status"] == "exited"
    assert mock_place.call_count == 3
    mock_get.assert_not_called()  # the successful attempt returned "filled" directly, no polling needed
    assert mock_discord.called  # at least the first failure got alerted


def test_unconfirmed_fill_is_canceled_and_retried():
    # First attempt submits, never reaches a terminal status within the
    # poll budget -> must cancel that stale order and try again (a fresh
    # order id) rather than declaring the position closed.
    with patch.object(oem, "time") as mock_time, \
         patch.object(oem, "latest_quote", return_value=100.0), \
         patch.object(oem, "latest_option_quote", return_value={"bid": 3.9, "ask": 4.1}), \
         patch.object(oem, "place_option_order", side_effect=[
             {"id": "stale-1", "status": "pending_new"},
             {"id": "o2", "status": "filled", "filled_avg_price": "4.0", "filled_qty": "1"},
         ]) as mock_place, \
         patch.object(oem, "get_order", return_value={"status": "pending_new"}), \
         patch.object(oem, "cancel_order") as mock_cancel, \
         patch.object(oem, "post_discord"):
        mock_time.sleep.side_effect = _no_sleep
        mock_time.time.side_effect = list(range(30))

        result = oem.watch_and_exit(
            underlying_symbol="TSLA", contract_symbol="TSLA260101C00100000",
            qty=1, option_side="call", entry_price=10.0,
        )

    assert result["status"] == "exited"
    assert mock_place.call_count == 2
    mock_cancel.assert_called_once_with("stale-1")
