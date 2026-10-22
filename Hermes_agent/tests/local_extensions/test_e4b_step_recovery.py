"""Source-linked recovery contracts, isolated banks and in-process service dispatch only."""
import copy
import json
from unittest.mock import patch

import pytest

from local_extensions.e4b_memory.cli import _save_step
from local_extensions.e4b_memory.service_runtime import Runtime, create
from local_extensions.e4b_memory.step_recovery import MAX_SESSION_STEPS
from local_extensions.e4b_memory.store import Store
from local_extensions.e4b_memory.surface import Surface


def step(step_id="save-check", revision=0):
    return {"step_id": step_id, "expected_revision": revision, "status": "blocked", "events": [
        {"id": "u1", "role": "user", "content": "Keep checkpoint 120; do not restart training.",
         "tool_call_id": "", "outcome": "unknown"},
        {"id": "t1", "role": "tool", "content": "Save of checkpoint 128 failed: read-only SSD.",
         "tool_call_id": "save-128", "outcome": "reported_failure"}], "facts": [
        {"category": "requirement", "event_id": "u1", "quote": "Keep checkpoint 120; do not restart training."},
        {"category": "observation", "event_id": "t1", "quote": "checkpoint 128 failed"}]}


def dispatch(runtime, token, action, values, session="s", instance="i"):
    return runtime.dispatch(token, {"session": session, "instance": instance,
                                    "action": action, "arguments": values})


def operation(surface, name, arguments=None):
    surface.dispatch({"action": "catalog", "section": "recovery"})
    surface.dispatch({"action": "describe", "section": "recovery", "name": name})
    return surface.dispatch({"action": "call", "section": "recovery", "name": name,
                             "arguments": arguments or {}})


def test_checkpoint_survives_restart_without_promoting_or_authorizing(tmp_path):
    store = Store(tmp_path)
    receipt = store.save_step("owner:session", step(), now=100)
    restarted = Store(tmp_path)
    saved = restarted.step_recovery("owner:session", "save-check")
    assert saved["events"] == step()["events"]
    assert saved["facts"] == step()["facts"]
    assert receipt["verified"] is False and saved["status_source"] == "controller_reported"
    assert restarted.step_recovery("other")["steps"] == []
    assert restarted.snapshot()["records"] == []
    assert restarted.checkpoint("owner:session", now=100) is None
    with pytest.raises(PermissionError):
        restarted.append("owner:session", layer="L1", kind="success", label="saved", content="128 saved",
                         evidence_ids=["t1"], now=100)


def test_retries_are_idempotent_and_conflicting_or_stale_writes_fail(tmp_path):
    store = Store(tmp_path)
    first = store.save_step("s", step(), now=100)
    revision = store.snapshot()["revision"]
    assert store.save_step("s", step(), now=200) == first
    assert store.snapshot()["revision"] == revision
    changed = step()
    changed["status"] = "completed"
    with pytest.raises(PermissionError):
        store.save_step("s", changed)
    with pytest.raises(PermissionError):
        store.save_step("s", step("second", 0))
    store.save_step("s", step("second", 1))
    assert store.step_recovery("s")["revision"] == 2


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(reasoning="hidden chain"),
    lambda p: p.update(verified=True),
    lambda p: p.update(expected_revision=True),
    lambda p: p["events"][0].update(role="assistant"),
    lambda p: p["events"][1].update(image="bytes"),
    lambda p: p["events"][1].update(content="data:image/png;base64,abc"),
    lambda p: p["events"][1].update(outcome="verified_success"),
    lambda p: p["events"][0].update(outcome="reported_success"),
    lambda p: p["facts"][1].update(quote="checkpoint 128 saved successfully"),
    lambda p: p["facts"][1].update(quote=" checkpoint 128 failed "),
    lambda p: p["facts"][1].update(event_id="missing"),
    lambda p: p["events"].append(copy.deepcopy(p["events"][0])),
    lambda p: p["events"][0].update(content="x" * 4001),
])
def test_invalid_or_unsupported_evidence_never_commits(tmp_path, mutation):
    store = Store(tmp_path)
    payload = step()
    mutation(payload)
    before = store.snapshot()
    with pytest.raises(ValueError):
        store.save_step("s", payload)
    assert store.snapshot() == before


def test_disk_failure_preserves_previous_checkpoint(tmp_path):
    store = Store(tmp_path)
    receipt = store.save_step("s", step())
    with patch("local_extensions.e4b_memory.store.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            store.save_step("s", step("second", 1))
    restarted = Store(tmp_path)
    assert restarted.step_recovery("s")["revision"] == 1
    assert restarted.step_recovery("s", "save-check")["digest"] == receipt["digest"]


def test_integrity_failure_and_readback_failure_are_not_success_receipts(tmp_path):
    store = Store(tmp_path)
    original_read = store._read
    reads = 0

    def corrupt_readback():
        nonlocal reads
        reads += 1
        state = original_read()
        if reads == 2:
            state["step_recovery"]["s"]["steps"][0]["events"][0]["content"] = "corrupted"
        return state

    with patch.object(store, "_read", side_effect=corrupt_readback):
        with pytest.raises(OSError, match="readback"):
            store.save_step("s", step())
    # A lost acknowledgement does not imply a lost commit: retry safely, never replay tools.
    assert Store(tmp_path).save_step("s", step())["revision"] == 1
    with store.transaction() as state:
        state["step_recovery"]["s"]["steps"][0]["events"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="integrity"):
        store.step_recovery("s", "save-check")


def test_aggregate_budget_refuses_intact_record_and_preserves_exact_source(tmp_path):
    store = Store(tmp_path)
    payload = step()
    payload["events"][0]["content"] = "  Keep exact units: -0.25 V, d > 0.\n"
    payload["facts"][0]["quote"] = payload["events"][0]["content"]
    store.save_step("s", payload)
    assert store.step_recovery("s", "save-check")["events"][0]["content"] == payload["events"][0]["content"]
    large = step("large", 1)
    large["facts"] *= 32
    large["facts"] = large["facts"][:32]
    large["events"] += [{"id": str(i), "role": "user", "content": "x" * 3900,
                         "tool_call_id": "", "outcome": "unknown"} for i in range(4)]
    with pytest.raises(ValueError, match="budget"):
        store.save_step("s", large)
    assert store.step_recovery("s")["revision"] == 1


def test_step_capacity_refuses_without_dropping_prior_evidence(tmp_path):
    store = Store(tmp_path)
    for index in range(MAX_SESSION_STEPS):
        store.save_step("s", step(str(index), index))
    before = store.snapshot()
    with pytest.raises(ValueError, match="no history was dropped"):
        store.save_step("s", step("overflow", MAX_SESSION_STEPS))
    assert store.snapshot() == before


def test_recovery_retention_defers_while_active(tmp_path):
    store = Store(tmp_path)
    store.save_step("s", step(), now=100)
    store.activity("s", True, now=101)
    store.expire(now=100 + 8 * 86400)
    assert store.step_recovery("s")["steps"]
    store.activity("s", False, now=102)
    store.expire(now=100 + 8 * 86400)
    assert store.step_recovery("s")["steps"] == []


def test_service_separates_roles_clients_and_requires_active_controller(tmp_path):
    runtime = create(tmp_path / "service")
    tokens = runtime.credentials.issue()
    dispatch(runtime, tokens["controller"], "begin", {"message": "Task"})
    with pytest.raises(PermissionError):
        dispatch(runtime, tokens["model"], "save_step", step())
    dispatch(runtime, tokens["controller"], "save_step", step())
    other = runtime.credentials.issue()
    dispatch(runtime, other["controller"], "begin", {"message": "Other task"})
    assert runtime.store.step_recovery(other["client_id"] + ":s")["steps"] == []
    with pytest.raises(PermissionError):
        dispatch(runtime, tokens["controller"], "save_step", step(), instance="not-started")
    restarted = Runtime(runtime.root)
    with pytest.raises(PermissionError):
        dispatch(restarted, tokens["controller"], "save_step", step())
    assert restarted.store.step_recovery(tokens["client_id"] + ":s")["revision"] == 1


def test_model_reads_only_discovered_scoped_steps_with_existing_budget(tmp_path):
    store = Store(tmp_path)
    store.save_step("s", step())
    surface = Surface(store, "s")
    with pytest.raises(ValueError):
        surface.dispatch({"action": "call", "section": "recovery", "name": "step",
                          "arguments": {"id": "save-check"}})
    assert operation(surface, "steps")["steps"][0]["verified"] is False
    for _ in range(3):
        assert operation(surface, "step", {"id": "save-check"})["status"] == "blocked"
    with pytest.raises(ValueError, match="budget"):
        operation(surface, "step", {"id": "save-check"})
    with pytest.raises(ValueError):
        operation(Surface(store, "other"), "step", {"id": "save-check"})


def test_owner_file_import_rejects_duplicate_json_fields(tmp_path):
    from types import SimpleNamespace
    store = Store(tmp_path / "bank")
    path = tmp_path / "step.json"
    path.write_text(json.dumps(step()), encoding="utf-8")
    args = SimpleNamespace(session="s", step_file=str(path))
    assert _save_step(store, args)["revision"] == 1
    path.write_text('{"step_id":"first","step_id":"second"}', encoding="utf-8")
    with pytest.raises(ValueError):
        _save_step(store, args)
