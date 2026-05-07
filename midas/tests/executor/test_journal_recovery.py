"""
tests/executor/test_journal_recovery.py

Tests for the journal infrastructure: serialisation, replay, and recovery.
Uses the journal's own data types directly — no planner or state machine needed.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from executor.fake_venue_adapter import FakeVenueAdapter
from executor.journal import (
    ExecutorJournal,
    InMemoryJournalStorage,
    JSONLFileJournalStorage,
    JournalEventSerializer,
    JournalEventType,
    JournalRecord,
    JournalReplayLoader,
    JournalWriter,
    JournalWriterConfig,
    RecoveryCoordinator,
    partition_key_from_ts_ns,
)


class JournalSerialiserTests(unittest.TestCase):
    """Tests that records survive a round-trip through JSON serialisation."""

    def test_opportunity_accepted_roundtrip(self) -> None:
        serializer = JournalEventSerializer()
        record = JournalRecord(
            event_id="evt-1",
            ts_ns=1_700_000_000_000_000_000,
            event_type=JournalEventType.OPPORTUNITY_ACCEPTED,
            package_id="pkg-1",
            opportunity_id="opp-1",
            relation_id="rel-1",
            payload={"metadata": {"expected_edge_bps": 50}},
        )

        encoded = serializer.dumps(record)
        decoded = serializer.loads(encoded)

        self.assertEqual(decoded.event_id, record.event_id)
        self.assertEqual(decoded.event_type, record.event_type)
        self.assertEqual(decoded.package_id, record.package_id)
        self.assertEqual(decoded.payload, record.payload)

    def test_opportunity_rejected_roundtrip(self) -> None:
        serializer = JournalEventSerializer()
        record = JournalRecord(
            event_id="evt-2",
            ts_ns=1_700_000_000_000_000_001,
            event_type=JournalEventType.OPPORTUNITY_REJECTED,
            package_id=None,
            opportunity_id="opp-2",
            relation_id="rel-2",
            payload={"reason": "trading halted"},
        )

        encoded = serializer.dumps(record)
        decoded = serializer.loads(encoded)

        self.assertEqual(decoded.event_type, JournalEventType.OPPORTUNITY_REJECTED)
        self.assertEqual(decoded.payload["reason"], "trading halted")
        self.assertIsNone(decoded.package_id)

    def test_kill_switch_roundtrip(self) -> None:
        serializer = JournalEventSerializer()
        record = JournalRecord(
            event_id="evt-ks",
            ts_ns=1_700_000_000_000_000_002,
            event_type=JournalEventType.KILL_SWITCH_ACTIVATED,
            package_id=None,
            opportunity_id=None,
            relation_id=None,
            payload={"reason": "daily loss cap", "automatic": True},
        )

        encoded = serializer.dumps(record)
        decoded = serializer.loads(encoded)

        self.assertEqual(decoded.event_type, JournalEventType.KILL_SWITCH_ACTIVATED)
        self.assertEqual(decoded.payload["automatic"], True)


class JournalWriterTests(unittest.TestCase):
    """Tests that the writer correctly persists records and supports replay."""

    def test_background_writer_flushes_all_records(self) -> None:
        storage = InMemoryJournalStorage()
        writer = JournalWriter(
            storage,
            config=JournalWriterConfig(background=True, flush_every_n_events=1),
        )
        journal = ExecutorJournal(writer)

        journal.record_opportunity_accepted(
            package_id="pkg-a",
            opportunity_id="opp-a",
            relation_id="rel-a",
            ts_ns=1_000_000_000_000,
        )
        journal.record_kill_switch_activation(
            reason="test halt",
            ts_ns=1_000_000_000_001,
            automatic=False,
        )
        writer.flush()
        writer.close()

        records = JournalReplayLoader(storage).load_records()
        self.assertEqual(len(records), 2)
        types = [r.event_type for r in records]
        self.assertIn(JournalEventType.OPPORTUNITY_ACCEPTED, types)
        self.assertIn(JournalEventType.KILL_SWITCH_ACTIVATED, types)

    def test_sync_writer_persists_immediately(self) -> None:
        storage = InMemoryJournalStorage()
        writer = JournalWriter(
            storage,
            config=JournalWriterConfig(background=False, flush_every_n_events=1),
        )
        journal = ExecutorJournal(writer)

        journal.record_opportunity_rejected(
            opportunity_id="opp-b",
            relation_id="rel-b",
            reason="max notional reached",
            ts_ns=2_000_000_000_000,
        )
        writer.close()

        records = JournalReplayLoader(storage).load_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event_type, JournalEventType.OPPORTUNITY_REJECTED)
        self.assertEqual(records[0].payload["reason"], "max notional reached")

    def test_jsonl_file_storage_writes_and_reads_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = JSONLFileJournalStorage(Path(tmpdir) / "journal")
            writer = JournalWriter(
                storage,
                config=JournalWriterConfig(background=False, flush_every_n_events=1),
            )
            journal = ExecutorJournal(writer)

            ts = 3_000_000_000_000
            journal.record_opportunity_accepted(
                package_id="pkg-file",
                opportunity_id="opp-file",
                relation_id="rel-file",
                ts_ns=ts,
            )
            writer.close()

            read_storage = JSONLFileJournalStorage(Path(tmpdir) / "journal")
            records = JournalReplayLoader(read_storage).load_records()

            self.assertEqual(len(records), 1)
            partition = partition_key_from_ts_ns(ts)
            self.assertIn(partition, read_storage.list_partitions())

    def test_replay_loader_filters_by_event_type(self) -> None:
        storage = InMemoryJournalStorage()
        writer = JournalWriter(
            storage,
            config=JournalWriterConfig(background=False, flush_every_n_events=1),
        )
        journal = ExecutorJournal(writer)

        journal.record_opportunity_accepted(
            package_id="pkg-filter",
            opportunity_id="opp-filter",
            relation_id="rel-filter",
            ts_ns=1_000,
        )
        journal.record_opportunity_rejected(
            opportunity_id="opp-filter-2",
            relation_id="rel-filter",
            reason="halted",
            ts_ns=2_000,
        )
        journal.record_kill_switch_activation(
            reason="test", ts_ns=3_000, automatic=True
        )
        writer.close()

        accepted_only = JournalReplayLoader(storage).load_records(
            event_types={JournalEventType.OPPORTUNITY_ACCEPTED}
        )
        self.assertEqual(len(accepted_only), 1)
        self.assertEqual(
            accepted_only[0].event_type, JournalEventType.OPPORTUNITY_ACCEPTED
        )


class RecoveryCoordinatorTests(unittest.TestCase):
    """Tests that recovery correctly loads journal state and reconciles with venue."""

    def test_empty_journal_recovers_cleanly(self) -> None:
        storage = InMemoryJournalStorage()
        venue = FakeVenueAdapter()
        recovery = RecoveryCoordinator(
            loader=JournalReplayLoader(storage)
        ).recover(venue_adapter=venue)

        self.assertEqual(recovery.records_loaded, 0)
        self.assertEqual(len(recovery.reconstructed_state.active_packages), 0)
        self.assertEqual(len(recovery.reconstructed_state.terminal_packages), 0)
        self.assertEqual(recovery.actions, tuple())

    def test_kill_switch_state_survives_replay(self) -> None:
        storage = InMemoryJournalStorage()
        writer = JournalWriter(
            storage,
            config=JournalWriterConfig(background=False, flush_every_n_events=1),
        )
        journal = ExecutorJournal(writer)
        journal.record_kill_switch_activation(
            reason="daily loss cap breached",
            ts_ns=1_000_000,
            automatic=True,
        )
        writer.close()

        records = JournalReplayLoader(storage).load_records()
        from executor.journal import JournalStateReconstructor
        state = JournalStateReconstructor().reconstruct(records)

        self.assertTrue(state.kill_switch_active)
        self.assertEqual(state.kill_switch_reason, "daily loss cap breached")


if __name__ == "__main__":
    unittest.main()
