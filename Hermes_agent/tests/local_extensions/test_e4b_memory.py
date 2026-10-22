"""Model-free behavioral contracts; run via scripts/run_tests.sh."""
import argparse
import concurrent.futures
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_extensions.e4b_memory import E4BMemoryProvider
from local_extensions.e4b_memory.cli import register_cli
from local_extensions.e4b_memory.store import Store
from local_extensions.e4b_memory.surface import Surface


class MemoryContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.evidence = [{"id": "verified-1", "kind": "success", "text": "Checksum-matched restore verified by owner."}]
        self.record = dict(layer="L1", kind="success", label="restore", content="Restore the checked backup.",
                           evidence_ids=["verified-1"])

    def grant(self, session="session-a", now=1000, budget=8):
        self.store.open_checkpoint(session, self.evidence, now=now, budget=budget)

    def test_append_requires_live_controller_grant_for_exact_session(self):
        with self.assertRaises(PermissionError):
            self.store.append("session-a", **self.record, now=1000)
        self.grant()
        with self.assertRaises(PermissionError):
            self.store.append("session-b", **self.record, now=1001)
        with self.assertRaises(PermissionError):
            self.store.append("session-a", **self.record, now=1300)
        self.assertEqual(self.store.snapshot()["records"], [])

    def test_budget_replay_and_restart(self):
        self.grant(budget=2)
        key = self.store.append("session-a", **self.record, now=1001)
        restarted = Store(self.temp.name)
        self.assertEqual(key, restarted.append("session-a", **self.record, now=1002))
        self.assertEqual(restarted.snapshot()["stages"]["session-a"]["remaining"], 1)
        restarted.append("session-a", **{**self.record, "content": "Second verified result."}, now=1003)
        with self.assertRaises(PermissionError):
            restarted.append("session-a", **{**self.record, "content": "Third result."}, now=1004)

    def test_only_approved_evidence_and_supported_kinds(self):
        self.grant()
        for changes, error in [({"kind": "routine_failure"}, ValueError),
                               ({"kind": "critical_failure"}, PermissionError),
                               ({"evidence_ids": ["invented"]}, PermissionError),
                               ({"layer": "L3"}, ValueError),
                               ({"content": "data:image/png;base64,AAAA"}, ValueError)]:
            with self.subTest(changes=changes), self.assertRaises(error):
                self.store.append("session-a", **{**self.record, **changes}, now=1001)
        self.assertEqual(self.store.snapshot()["records"], [])

    def test_verified_critical_lesson_and_explicit_preference(self):
        self.store.open_checkpoint("a", [
            {"id": "limit", "kind": "critical_failure", "text": "Owner confirmed driver lacks the required feature."},
            {"id": "pref", "kind": "preference", "text": "Owner explicitly requested concise replies."}], now=1000)
        self.store.append("a", layer="L1", kind="critical_failure", label="driver", content="Feature unsupported.",
                          evidence_ids=["limit"], now=1001)
        self.store.append("a", layer="L3", kind="preference", label="style", content="Concise replies.",
                          evidence_ids=["pref"], now=1002)
        self.assertEqual(len(self.store.snapshot()["records"]), 2)

    def test_consolidation_waits_for_idle_and_all_turns(self):
        self.grant()
        key = self.store.append("session-a", **self.record, now=1001)
        self.assertEqual(self.store.consolidate(now=11800), 0)
        self.store.activity("busy", True, now=1002)
        self.assertEqual(self.store.consolidate(now=20000), 0)
        self.store.activity("busy", False, now=20001)
        before = self.store.snapshot()["records"][0]
        self.assertEqual(self.store.consolidate(now=30801), 1)
        after = Store(self.temp.name).snapshot()["records"][0]
        self.assertEqual(after["id"], key)
        self.assertEqual(after["content"], before["content"])
        self.assertEqual(after["evidence"], before["evidence"])
        self.assertEqual(after["bucket"], "persistent")
        self.assertEqual(self.store.consolidate(now=50000), 0)

    def test_failed_commit_keeps_temporary_records(self):
        self.grant()
        self.store.append("session-a", **self.record, now=1001)
        with patch("local_extensions.e4b_memory.store.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.store.consolidate(now=20000)
        self.assertEqual(Store(self.temp.name).snapshot()["records"][0]["bucket"], "temporary")
        self.assertEqual(self.store.consolidate(now=20001), 1)

    def test_parallel_appends_do_not_lose_records(self):
        self.grant()
        def append(index):
            return Store(self.temp.name).append("session-a", **{**self.record, "content": f"Verified result {index}"}, now=1001)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(append, range(8)))
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(len(self.store.snapshot()["records"]), 8)

    def test_staged_discovery_and_section_scoped_reads(self):
        self.grant()
        key = self.store.append("session-a", **self.record, now=1001)
        surface = Surface(self.store, "session-a")
        initial = surface.dispatch({"action": "sections"})
        self.assertIn("temporary.L1", initial["sections"])
        with self.assertRaises(ValueError):
            surface.dispatch({"action": "call", "section": "temporary.L1", "name": "read", "arguments": {"id": key}})
        catalog = surface.dispatch({"action": "catalog", "section": "temporary.L1"})
        self.assertNotIn("parameters", catalog["operations"][0])
        surface.dispatch({"action": "describe", "section": "temporary.L1", "name": "read"})
        result = surface.dispatch({"action": "call", "section": "temporary.L1", "name": "read", "arguments": {"id": key}})
        self.assertEqual(result["record"]["content"], self.record["content"])
        surface.dispatch({"action": "catalog", "section": "persistent.L1"})
        surface.dispatch({"action": "describe", "section": "persistent.L1", "name": "read"})
        with self.assertRaises(ValueError):
            surface.dispatch({"action": "call", "section": "persistent.L1", "name": "read", "arguments": {"id": key}})

    def test_no_model_edit_delete_or_grant_operations(self):
        surface = Surface(self.store, "session-a")
        surface.dispatch({"action": "catalog", "section": "checkpoint"})
        for name in ("edit", "delete", "open_checkpoint", "consolidate"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                surface.dispatch({"action": "describe", "section": "checkpoint", "name": name})

    def test_owner_can_edit_and_delete(self):
        self.grant()
        key = self.store.append("session-a", **self.record, now=1001)
        self.store.owner_edit(key, content="Owner correction")
        self.assertEqual(self.store.snapshot()["records"][0]["content"], "Owner correction")
        self.store.owner_edit(key, delete=True)
        self.assertEqual(self.store.snapshot()["records"], [])

    def test_malformed_storage_is_not_replaced(self):
        self.store.path.write_text("broken JSON", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.activity("session-a", True)
        self.assertEqual(self.store.path.read_text(encoding="utf-8"), "broken JSON")

    def test_provider_refuses_gateway_and_child_contexts(self):
        for context in ({"platform": "telegram"}, {"platform": "cli", "agent_context": "subagent"}):
            with self.subTest(context=context), self.assertRaises(PermissionError):
                E4BMemoryProvider().initialize("a", hermes_home=self.temp.name, **context)

    def test_provider_turn_accounting_handles_delayed_sync(self):
        provider = E4BMemoryProvider()
        provider.initialize("a", hermes_home=self.temp.name, platform="cli")
        self.addCleanup(provider.shutdown)
        provider.on_turn_start(1, "first")
        provider.on_turn_start(2, "next turn before queued sync")
        provider.sync_turn("first", "done", session_id="a")
        self.assertTrue(provider.store.snapshot()["active"])
        provider.sync_turn("second", "done", session_id="a")
        self.assertEqual(provider.store.snapshot()["active"], {})
        self.assertEqual(provider.store.snapshot()["records"], [])

    def test_owner_cli_parses_without_model_commands(self):
        parser = argparse.ArgumentParser()
        register_cli(parser)
        args = parser.parse_args(["checkpoint", "session-a", "evidence.json"])
        self.assertEqual(args.session, "session-a")
        self.assertTrue(callable(args.func))


def test_real_memory_provider_discovery():
    """Exercise the supported user-plugin discovery path, not a fake provider loader."""
    import shutil
    from plugins.memory import find_provider_dir, load_memory_provider
    from hermes_constants import get_hermes_home
    plugin = get_hermes_home() / "plugins" / "e4b_memory"
    source = Path(__file__).resolve().parents[2] / "local_extensions" / "e4b_memory"
    shutil.copytree(source.resolve(), plugin)
    assert find_provider_dir("e4b_memory") == plugin
    provider = load_memory_provider("e4b_memory", register_skills=False)
    assert provider is not None
    assert provider.name == "e4b_memory"
    assert provider.get_tool_schemas()[0]["name"] == "e4b_memory"
