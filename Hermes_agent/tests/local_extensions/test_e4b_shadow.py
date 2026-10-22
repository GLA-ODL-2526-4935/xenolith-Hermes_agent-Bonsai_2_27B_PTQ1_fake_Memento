"""Offline shadow/recovery contracts; no application, server, model or GPU."""
import argparse
import copy
from unittest.mock import patch

import pytest

from local_extensions.e4b_memory import E4BMemoryProvider
from local_extensions.e4b_memory.checkpoint import (
    capture_shadow, preview_rebuild, read_checkpoint, require_checkpoint, try_shadow,
)
from local_extensions.e4b_memory.cli import COMMANDS, register_cli
from local_extensions.e4b_memory.service_runtime import create
from local_extensions.e4b_memory.store import Store
from local_extensions.e4b_memory.task_state import annotate_task, task_view
from local_extensions.e4b_memory.surface import Surface


def transcript():
    return [{"role": "system", "content": "Use the selected tools only."},
            {"role": "user", "content": "Keep checkpoint 120; do not restart training."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"success":false,"error":"Read-only SSD"}'},
            {"role": "assistant", "content": "The save did not complete."}]


def test_lossless_shadow_restart_preview_and_no_permission(tmp_path):
    store = Store(tmp_path)
    messages = transcript()
    before = copy.deepcopy(messages)
    saved = capture_shadow(store, "s", messages, now=100)
    assert messages == before
    restarted = Store(tmp_path)
    assert require_checkpoint(restarted, "s", messages) == saved
    preview = preview_rebuild(restarted, "s")
    assert preview["messages"] == messages
    assert preview["live_context_changed"] is False
    assert preview["shortening_enabled"] is False
    assert saved["context_replacement_allowed"] is False
    assert restarted.checkpoint("s", now=100) is None
    assert restarted.snapshot()["records"] == []
    assert read_checkpoint(restarted, "other") is None
    items = preview["task_state"]["items"]
    assert items[0]["source"]["content"] == messages[1]["content"]
    assert items[1]["source"]["outcome"] == "reported_failure"
    assert all(item["status"] == "pending" for item in items)


def test_append_retry_and_rewind_refusal(tmp_path):
    store, messages = Store(tmp_path), transcript()
    first = capture_shadow(store, "s", messages)
    revision = store.snapshot()["revision"]
    assert capture_shadow(store, "s", messages) == first
    assert store.snapshot()["revision"] == revision
    messages.append({"role": "user", "content": "Use checkpoint 128 instead; still do not restart."})
    second = capture_shadow(store, "s", messages)
    assert second["revision"] == 2
    assert len(store.step_recovery("s")["steps"]) == 3
    with pytest.raises(PermissionError, match="rewound"):
        capture_shadow(store, "s", transcript())
    with pytest.raises(PermissionError):
        require_checkpoint(store, "s", transcript())


def test_partial_commit_reconciles_without_replaying_tools(tmp_path):
    store = Store(tmp_path)
    messages = [{"role": "user", "content": "Keep exact value -0.25 V."}]
    original = store._write
    count = 0

    def fail_manifest(state):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("disk full")
        original(state)

    with patch.object(store, "_write", side_effect=fail_manifest):
        with pytest.raises(OSError):
            capture_shadow(store, "s", messages)
    restarted = Store(tmp_path)
    assert read_checkpoint(restarted, "s") is None
    assert len(restarted.step_recovery("s")["steps"]) == 1
    with pytest.raises(PermissionError):
        require_checkpoint(restarted, "s", messages)
    assert capture_shadow(restarted, "s", messages)["readback"] is True
    assert len(restarted.step_recovery("s")["steps"]) == 1


@pytest.mark.parametrize("messages", [
    [], [{"role": "user", "content": [{"type": "image_url", "image_url": "image"}]}],
    [{"role": "assistant", "content": "visible", "reasoning_content": "hidden"}],
    [{"role": "tool", "tool_call_id": "missing", "content": "orphan"}],
    [{"role": "user", "content": "x" * 4001}],
    [{"role": "user", "content": "password=verysecret123"}],
    [{"role": "user", "content": "request"}] * 201,
])
def test_refused_capture_has_no_success_manifest_or_source_truncation(tmp_path, messages):
    store = Store(tmp_path)
    result = try_shadow(store, "s", messages)
    assert result["readback"] is False
    assert result["context_replacement_allowed"] is False
    assert read_checkpoint(store, "s") is None


def test_pending_tool_envelope_saved_but_cannot_pass_future_gate(tmp_path):
    messages = transcript()[:3]
    store = Store(tmp_path)
    capture_shadow(store, "s", messages)
    assert preview_rebuild(store, "s")["pending_tool_calls"] == ["c1"]
    with pytest.raises(PermissionError, match="Pending"):
        require_checkpoint(store, "s", messages)
    capture_shadow(store, "s", transcript())
    assert require_checkpoint(store, "s", transcript())["readback"] is True


def test_changed_or_expired_evidence_invalidates_checkpoint(tmp_path):
    store = Store(tmp_path)
    capture_shadow(store, "s", transcript(), now=100)
    with store.transaction() as state:
        state["shadow_checkpoints"]["s"]["messages"][1]["content"] = "changed"
    with pytest.raises(ValueError, match="integrity"):
        read_checkpoint(store, "s")


def test_retention_removes_manifest_when_old_source_expires(tmp_path):
    store = Store(tmp_path)
    capture_shadow(store, "s", transcript(), now=100)
    more = transcript() + [{"role": "user", "content": "Continue the same task."}]
    capture_shadow(store, "s", more, now=100 + 6 * 86400)
    store.expire(now=100 + 8 * 86400)
    assert read_checkpoint(store, "s") is None
    assert len(store.step_recovery("s")["steps"]) == 1


def test_owner_status_and_explicit_supersession_are_not_automatic(tmp_path):
    store = Store(tmp_path)
    messages = transcript() + [{"role": "user", "content": "Use checkpoint 128 instead."}]
    capture_shadow(store, "s", messages)
    view = task_view(store, "s")
    first, tool, last = view["items"]
    assert first["superseded_by"] is None
    update = {"id": first["id"], "status": "cancelled", "superseded_by": last["id"],
              "note": "Owner replaced the old checkpoint requirement.", "expected_revision": 0}
    assert annotate_task(store, "s", update)["verified"] is False
    view = task_view(Store(tmp_path), "s")
    assert view["items"][0]["superseded_by"] == last["id"]
    assert view["items"][0]["source"]["content"] == messages[1]["content"]
    with pytest.raises(PermissionError, match="Stale"):
        annotate_task(store, "s", update)
    with pytest.raises(ValueError, match="Supersession"):
        annotate_task(store, "s", {**update, "superseded_by": tool["id"], "expected_revision": 1})


def test_provider_hooks_capture_opt_in_without_changing_transcript(tmp_path):
    provider = E4BMemoryProvider()
    with patch("local_extensions.e4b_memory.load_settings", return_value={"shadow_capture": True}):
        provider.initialize("s", platform="cli", hermes_home=tmp_path)
    try:
        provider.on_turn_start(1, "Task")
        provider.sync_turn("Task", "Answer", messages=transcript())
        assert provider.shadow_status["readback"] is True
        assert provider.store.snapshot()["active"] == {}
        assert provider.pre_compress_checkpoint_api_version == 1
        provider.on_pre_compress(transcript())
        assert provider.store.step_recovery("s")["revision"] == 2
    finally:
        provider.shutdown()


def test_service_shadow_is_controller_only_and_owner_cli_commands_parse(tmp_path):
    runtime = create(tmp_path / "service")
    tokens = runtime.credentials.issue()

    def dispatch(role, action, values):
        return runtime.dispatch(tokens[role], {"session": "s", "instance": "i", "action": action, "arguments": values})

    dispatch("controller", "begin", {"message": "Task"})
    with pytest.raises(PermissionError):
        dispatch("model", "shadow", {"messages": transcript()})
    assert dispatch("controller", "shadow", {"messages": transcript()})["readback"] is True
    dispatch("controller", "finish", {"candidates": [], "error": ""})
    with pytest.raises(PermissionError):
        dispatch("controller", "shadow", {"messages": transcript()})
    parser = argparse.ArgumentParser()
    register_cli(parser)
    for command in ("task-state", "preview-context"):
        args = parser.parse_args([command, tokens["client_id"] + ":s"])
        assert isinstance(COMMANDS[command](runtime.store, args), dict)


def test_task_discovery_is_scoped_and_shares_expansion_budget(tmp_path):
    store = Store(tmp_path)
    capture_shadow(store, "s", transcript())
    surface = Surface(store, "s")
    surface.dispatch({"action": "catalog", "section": "recovery"})
    surface.dispatch({"action": "describe", "section": "recovery", "name": "task_items"})
    result = surface.dispatch({"action": "call", "section": "recovery", "name": "task_items"})
    item_id = result["items"][0]["id"]
    assert "source" not in result["items"][0]
    surface.dispatch({"action": "describe", "section": "recovery", "name": "task_item"})
    args = {"action": "call", "section": "recovery", "name": "task_item", "arguments": {"id": item_id}}
    for _ in range(3):
        assert surface.dispatch(args)["grants_permission"] is False
    with pytest.raises(ValueError, match="budget"):
        surface.dispatch(args)
    other = Surface(store, "other")
    other.dispatch({"action": "catalog", "section": "recovery"})
    other.dispatch({"action": "describe", "section": "recovery", "name": "task_item"})
    with pytest.raises(ValueError, match="not found"):
        other.dispatch(args)


def test_generated_profile_shadow_opt_in_is_compatible_and_does_not_launch(tmp_path):
    from local_extensions.e4b_memory.restricted import create_profile, profile_config
    import json
    default = profile_config("http://127.0.0.1:8000/v1", "future-e4b")
    assert "shadow_capture" not in default["memory"]["e4b_memory"]
    path = create_profile(tmp_path / "profile", "http://127.0.0.1:8000/v1", "future-e4b", shadow_capture=True)
    assert json.loads((path / "config.yaml").read_text())["memory"]["e4b_memory"]["shadow_capture"] is True
    with pytest.raises(FileExistsError):
        create_profile(path, "http://127.0.0.1:8000/v1", "future-e4b")


def test_remote_provider_shadow_hook_uses_controller_not_model_client(tmp_path):
    from local_extensions.e4b_memory.remote import RemoteMemoryProvider
    runtime = create(tmp_path / "service")
    tokens = runtime.credentials.issue()

    class BoundClient:
        def __init__(self, role):
            self.role = role

        def call(self, session, instance, action, arguments):
            return runtime.dispatch(tokens[self.role], {"session": session, "instance": instance,
                                    "action": action, "arguments": arguments})

    provider = RemoteMemoryProvider()
    provider.session, provider.instance, provider.shadow_capture = "s", "i", True
    provider.model, provider.controller = BoundClient("model"), BoundClient("controller")
    provider.on_turn_start(1, "Task")
    provider.sync_turn("Task", "Answer", messages=transcript())
    assert provider.shadow_status["readback"] is True
    assert not hasattr(provider, "store")
    assert runtime.store.snapshot()["active"] == {}
