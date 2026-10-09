# Covers the restart-replay bug: the Patreon poller's "seen" set used to
# be in-memory only, so every restart reset it to empty and the next
# poll's most recent posts all looked brand new. Fixed via HybridStore's
# durable ingest_seen table (mark_ingest_seen/get_all_ingest_seen).
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lavish_core.db.hybrid_store import HybridStore
from lavish_core.patreon.patreon_trigger import _post_age_seconds


def test_mark_and_recall_seen_survives_a_fresh_store_instance(tmp_path):
    db_path = tmp_path / "test.duckdb"
    store1 = HybridStore(duckdb_path=db_path)
    store1.mark_ingest_seen("patreon", "post_1")
    store1.mark_ingest_seen("patreon", "post_2")

    # A brand new HybridStore instance against the same file - simulates
    # a process restart, where the old in-memory set would have reset.
    store2 = HybridStore(duckdb_path=db_path)
    seen = store2.get_all_ingest_seen("patreon")

    assert seen == {"post_1", "post_2"}


def test_marking_the_same_item_twice_is_idempotent(tmp_path):
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    store.mark_ingest_seen("patreon", "post_1")
    store.mark_ingest_seen("patreon", "post_1")  # must not raise (duplicate PK)
    assert store.get_all_ingest_seen("patreon") == {"post_1"}


def test_sources_are_isolated(tmp_path):
    store = HybridStore(duckdb_path=tmp_path / "test.duckdb")
    store.mark_ingest_seen("patreon", "post_1")
    store.mark_ingest_seen("discord", "msg_1")
    assert store.get_all_ingest_seen("patreon") == {"post_1"}
    assert store.get_all_ingest_seen("discord") == {"msg_1"}


def test_post_age_seconds_computes_a_real_age():
    ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat().replace("+00:00", "Z")
    post = {"attributes": {"created_at": ts}}
    age = _post_age_seconds(post)
    assert age is not None
    assert 110 <= age <= 130


def test_post_age_seconds_handles_missing_or_bad_timestamp():
    assert _post_age_seconds({"attributes": {}}) is None
    assert _post_age_seconds({"attributes": {"created_at": "not-a-date"}}) is None
