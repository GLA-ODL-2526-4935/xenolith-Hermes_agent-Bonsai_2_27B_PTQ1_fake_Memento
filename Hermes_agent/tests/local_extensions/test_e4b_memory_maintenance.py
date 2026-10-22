"""CPU-only contracts for automated review, retention and restricted launch policy."""
import json
from contextlib import redirect_stdout
import io
import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from local_extensions.e4b_memory import E4BMemoryProvider, load_settings
from local_extensions.e4b_memory.maintenance import DAY, extract_evidence
from local_extensions.e4b_memory.restricted import create_profile, network_policy, validate_surface
from local_extensions.e4b_memory.sorting import decode_response, local_endpoint, request_sort
from local_extensions.e4b_memory.store import Store
from local_extensions.e4b_memory.surface import Surface


class MaintenanceContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.messages = [{"role": "tool", "content": json.dumps({"success": True, "message": "Checksum matched"})}]

    def add(self, layer="L1", now=1000):
        self.store.open_checkpoint("a", [{"id": "proof", "kind": "success", "text": "Owner verified checksum."}], now=now)
        return self.store.append("a", layer=layer, kind="success", label="backup", content="Use the verified backup.",
                                 evidence_ids=["proof"], now=now + 1)

    def test_tool_results_are_only_review_candidates(self):
        self.store.task_start("a", "Restore the backup", now=1000)
        self.store.capture("a", self.messages, reason="context_pressure", now=1001)
        state = self.store.snapshot()
        self.assertFalse(state["requests"]["a"]["approved"])
        self.assertIsNone(self.store.checkpoint("a", now=1002))
        self.assertEqual(state["records"], [])
        key = state["requests"]["a"]["evidence"][0]["id"]
        self.store.approve_request("a", [key], now=1002)
        self.assertIn(key, self.store.checkpoint("a", now=1003)["evidence"])
        self.assertEqual(self.store.snapshot()["records"], [])

    def test_idle_checkpoint_waits_for_safe_boundary_and_deduplicates(self):
        self.store.task_start("a", "Restore", now=1000)
        self.store.capture("a", self.messages, now=1001)
        self.store.activity("busy", True, now=1001)
        self.store.prepare_idle_checkpoints(now=2000)
        self.assertEqual(self.store.snapshot()["requests"], {})
        self.store.activity("busy", False, now=2001)
        self.store.prepare_idle_checkpoints(now=2300)
        self.assertEqual(self.store.snapshot()["requests"], {})
        self.store.prepare_idle_checkpoints(now=2301)
        revision = self.store.snapshot()["revision"]
        self.store.prepare_idle_checkpoints(now=2400)
        self.assertEqual(self.store.snapshot()["revision"], revision)
        key = self.store.snapshot()["requests"]["a"]["evidence"][0]["id"]
        self.store.approve_request("a", [key], now=2401)
        self.store.prepare_idle_checkpoints(now=3000)
        self.assertEqual(self.store.snapshot()["requests"], {})

    def test_routine_failures_assistant_claims_and_media_are_not_candidates(self):
        messages = [{"role": "tool", "content": '{"success":false,"error":"Bad argument"}'},
                    {"role": "assistant", "content": "I definitely succeeded!", "reasoning": "private"},
                    {"role": "tool", "content": '{"success":true,"image_data":"encoded"}'},
                    {"role": "tool", "content": '{"success":true,"error":"still broken"}'}]
        candidates, error = extract_evidence(messages)
        self.assertEqual(candidates, [])
        self.assertEqual(error, "still broken")
        self.store.task_start("a", "Try a task", now=1000)
        self.store.capture("a", messages, now=1001)
        self.assertEqual(self.store.recovery("a")["last_error"], "still broken")
        self.assertIsNone(self.store.recovery("b"))
        self.assertEqual(self.store.snapshot()["records"], [])

    def test_memory_reads_do_not_become_new_evidence(self):
        messages = [{"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "e4b_memory"}}]},
                    {"role": "tool", "tool_call_id": "x", "content": '{"success":true,"record":{"content":"old"}}'}]
        self.assertEqual(extract_evidence(messages)[0], [])

    def test_critical_tool_failure_is_candidate_not_verified_memory(self):
        messages = [{"role": "tool", "content": '{"success":false,"critical_failure":true,"error":"Data corruption"}'}]
        candidates, error = extract_evidence(messages)
        self.assertEqual(candidates[0]["kind"], "critical_failure")
        self.assertEqual(error, "Data corruption")

    def test_recovery_is_on_demand_and_owner_can_complete(self):
        self.store.task_start("a", "Unfinished restore", now=1000)
        surface = Surface(self.store, "a")
        surface.dispatch({"action": "catalog", "section": "recovery"})
        surface.dispatch({"action": "describe", "section": "recovery", "name": "recovery"})
        task = surface.dispatch({"action": "call", "section": "recovery", "name": "recovery"})["task"]
        self.assertEqual(task["status"], "in_progress")
        self.store.complete_task("a")
        self.assertEqual(self.store.recovery("a")["status"], "complete")
        self.store.task_start("a", "New task", now=2000)
        self.assertEqual(self.store.recovery("a")["goal"], "New task")

    def test_retention_expires_only_persistent_l0(self):
        self.add("L0")
        self.store.consolidate(now=20000)
        self.add("L1", now=20001)
        self.assertEqual(self.store.expire(now=20000 + 30 * DAY), 1)
        remaining = self.store.snapshot()["records"]
        self.assertEqual(remaining[0]["layer"], "L1")
        self.assertEqual(remaining[0]["bucket"], "temporary")

    def test_retention_defers_while_busy_and_expires_recovery_separately(self):
        self.add("L0")
        self.store.consolidate(now=20000)
        self.store.task_start("a", "Unfinished", now=20001)
        self.store.activity("busy", True, now=20002)
        self.assertEqual(self.store.expire(now=40 * DAY), 0)
        self.assertIsNotNone(self.store.recovery("a"))
        self.store.activity("busy", False, now=40 * DAY)
        self.assertEqual(self.store.expire(now=40 * DAY), 1)
        self.assertIsNone(self.store.recovery("a"))

    def test_sorting_changes_layer_not_content(self):
        key = self.add()
        snapshot = self.store.maintenance_snapshot(now=20000)
        self.assertEqual(self.store.apply_sort_plan(snapshot["revision"], [{"id": key, "layer": "L2"}], now=20000), 1)
        record = self.store.snapshot()["records"][0]
        self.assertEqual(record["content"], "Use the verified backup.")
        self.assertEqual(record["layer"], "L2")
        self.assertEqual(record["bucket"], "persistent")

    def test_sorting_rejects_stale_results_and_text_edits(self):
        key = self.add()
        snapshot = self.store.maintenance_snapshot(now=20000)
        for assignment in ({"id": key, "layer": "L2", "content": "rewrite"},
                           {"id": key, "layer": "L3"}, {"id": "invented", "layer": "L1"}):
            with self.subTest(assignment=assignment), self.assertRaises(ValueError):
                self.store.apply_sort_plan(snapshot["revision"], [assignment], now=20000)
        self.store.activity("new-work", True, now=20001)
        with self.assertRaises(PermissionError):
            self.store.apply_sort_plan(snapshot["revision"], [{"id": key, "layer": "L2"}], now=20002)
        self.assertEqual(self.store.snapshot()["records"][0]["bucket"], "temporary")

    def test_owner_edit_invalidates_inflight_sort_plan(self):
        key = self.add()
        snapshot = self.store.maintenance_snapshot(now=20000)
        self.store.owner_edit(key, content="Owner correction")
        with self.assertRaises(PermissionError):
            self.store.apply_sort_plan(snapshot["revision"], [{"id": key, "layer": "L2"}], now=20000)

    def test_sorter_has_no_cloud_or_redirect_fallback(self):
        self.assertEqual(local_endpoint("http://localhost:8000/v1"), "http://127.0.0.1:8000/v1/chat/completions")
        for endpoint in ("https://example.com/v1", "http://192.168.1.2/v1", "http://127.0.0.1.evil/v1",
                         "http://user:pass@127.0.0.1/v1", "http://127.0.0.1/v1?target=remote"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                local_endpoint(endpoint)
        from local_extensions.e4b_memory.sorting import NoRedirect
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.com")

    def test_sorter_response_validation(self):
        reply = {"choices": [{"finish_reason": "stop", "message": {"content": '{"assignments":[]}'}}]}
        self.assertEqual(decode_response(json.dumps(reply)), [])
        for malformed in ('{}', '{"choices":[]}', '{"choices":[{"finish_reason":"length"}]}'):
            with self.subTest(reply=malformed), self.assertRaises(ValueError):
                decode_response(malformed)

    def test_sorter_failure_keeps_temporary_records(self):
        self.add()
        provider = E4BMemoryProvider()
        provider.store = self.store
        provider.settings = {"semantic_sort": True, "base_url": "http://127.0.0.1:8000/v1", "model": "same-main-model",
                             "l0_days": 30, "recovery_days": 7}
        with patch("local_extensions.e4b_memory.request_sort", side_effect=ValueError("malformed model reply")):
            with self.assertRaises(ValueError):
                provider.maintain()
        self.assertEqual(self.store.snapshot()["records"][0]["bucket"], "temporary")

    def test_profile_creation_never_overwrites_and_disables_builtin_memory(self):
        destination = Path(self.temp.name) / "profile"
        create_profile(destination, "http://127.0.0.1:8000/v1", "test-model", semantic_sort=True)
        config = json.loads((destination / "config.yaml").read_text())
        self.assertTrue(config["memory"]["e4b_memory"]["semantic_sort"])
        self.assertFalse(config["memory"]["memory_enabled"])
        self.assertEqual(config["platform_toolsets"]["cli"], ["memory"])
        with self.assertRaises(FileExistsError):
            create_profile(destination, "http://127.0.0.1:8000/v1", "test-model")

    def test_launcher_refuses_extra_tool_or_missing_provider(self):
        agent = SimpleNamespace(tools=[{"function": {"name": "e4b_memory"}}],
                                _memory_manager=SimpleNamespace(providers=[SimpleNamespace(name="e4b_memory")]))
        validate_surface(agent)
        agent.tools.append({"function": {"name": "terminal"}})
        with self.assertRaises(PermissionError):
            validate_surface(agent)
        agent.tools.pop()
        agent._memory_manager = None
        with self.assertRaises(PermissionError):
            validate_surface(agent)

    def test_network_guard_blocks_external_addresses(self):
        network_policy("socket.connect", (None, ("127.0.0.1", 8000)))
        network_policy("socket.getaddrinfo", ("localhost",))
        for event, args in (("socket.connect", (None, ("1.1.1.1", 443))),
                            ("socket.getaddrinfo", ("api.example.com",)), ("socket.connect", (None, "/var/run/docker.sock"))):
            with self.subTest(event=event, args=args), self.assertRaises(PermissionError):
                network_policy(event, args)

    def test_precompression_hook_prepares_review_not_persistent_memory(self):
        provider = E4BMemoryProvider()
        provider.store, provider.session = self.store, "a"
        provider.on_pre_compress(self.messages)
        self.assertIn("a", self.store.snapshot()["requests"])
        self.assertEqual(self.store.snapshot()["records"], [])

    def test_read_budget_resets_between_turns(self):
        key = self.add()
        surface = Surface(self.store, "a")
        surface.dispatch({"action": "catalog", "section": "temporary.L1"})
        surface.dispatch({"action": "describe", "section": "temporary.L1", "name": "read"})
        args = {"action": "call", "section": "temporary.L1", "name": "read", "arguments": {"id": key}}
        for _ in range(3):
            surface.dispatch(args)
        with self.assertRaises(ValueError):
            surface.dispatch(args)
        surface.reset_budget()
        self.assertEqual(surface.dispatch(args)["record"]["id"], key)

    def test_legacy_model_config_works_when_sorting_is_off(self):
        with patch("hermes_cli.config.load_config_readonly", return_value={"model": "legacy-model"}):
            self.assertFalse(load_settings()["semantic_sort"])
        config = {"model": "legacy-model", "memory": {"e4b_memory": {"semantic_sort": True}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=config), self.assertRaises(ValueError):
            load_settings()

    def test_configured_sorter_requires_the_explicit_local_main_model(self):
        config = {"model": {"provider": "custom", "base_url": "http://127.0.0.1:8000/v1", "default": "main-model"},
                  "memory": {"e4b_memory": {"semantic_sort": True}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=config):
            self.assertEqual(load_settings()["model"], "main-model")
        config["model"]["base_url"] = "https://remote.example/v1"
        with patch("hermes_cli.config.load_config_readonly", return_value=config), self.assertRaises(ValueError):
            load_settings()

    def test_sort_transport_uses_only_the_configured_model_and_text(self):
        key = self.add()
        snapshot = self.store.maintenance_snapshot(now=20000)
        reply = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"assignments": [{"id": key, "layer": "L2"}]})}}]}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(reply).encode()
        opener = MagicMock()
        opener.open.return_value = response
        with patch("local_extensions.e4b_memory.sorting.build_opener", return_value=opener):
            result = request_sort(snapshot, "http://127.0.0.1:8000/v1", "main-model")
        request = opener.open.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["model"], "main-model")
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(result, [{"id": key, "layer": "L2"}])
        self.assertNotIn("evidence", body["messages"][1]["content"])

    def test_complete_sort_plan_is_required(self):
        key = self.add()
        snapshot = self.store.maintenance_snapshot(now=20000)
        for assignments in ([], [{"id": key, "layer": "L1"}, {"id": key, "layer": "L2"}]):
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                self.store.apply_sort_plan(snapshot["revision"], assignments, now=20000)

    def test_owner_command_dispatch_against_an_isolated_profile(self):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from local_extensions.e4b_memory.cli import register_cli, run
        token = set_hermes_home_override(self.temp.name)
        self.addCleanup(reset_hermes_home_override, token)
        parser = argparse.ArgumentParser()
        register_cli(parser)
        evidence_file = Path(self.temp.name) / "evidence.json"
        evidence_file.write_text(json.dumps([{"id": "proof", "kind": "success", "text": "Owner verified."}]))
        with redirect_stdout(io.StringIO()):
            run(parser.parse_args(["checkpoint", "a", str(evidence_file)]))
        store = Store(Path(self.temp.name) / "e4b_memory")
        key = store.append("a", layer="L1", kind="success", label="test", content="Verified procedure.", evidence_ids=["proof"])
        replacement = Path(self.temp.name) / "replacement.txt"
        replacement.write_text("Owner corrected procedure.")
        with redirect_stdout(io.StringIO()):
            run(parser.parse_args(["edit", key, str(replacement)]))
        self.assertEqual(store.snapshot()["records"][0]["content"], "Owner corrected procedure.")
        with redirect_stdout(io.StringIO()):
            run(parser.parse_args(["delete", key]))
        self.assertEqual(store.snapshot()["records"], [])
