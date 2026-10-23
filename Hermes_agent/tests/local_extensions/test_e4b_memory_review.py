"""Offline behavior contracts: automatic proposals, manual authority and source preservation."""
import argparse
import copy
import io
import json
from types import SimpleNamespace

import pytest

from local_extensions.e4b_memory import E4BMemoryProvider, load_settings
from local_extensions.e4b_memory.cli import COMMANDS, register_cli
from local_extensions.e4b_memory.review import configure_review, run_review, decide, retry_failed
from local_extensions.e4b_memory.service import settings
from local_extensions.e4b_memory.service_runtime import create
from local_extensions.e4b_memory.store import Store
from local_extensions.e4b_memory.surface import Surface


CONFIG = {"automatic_review": True, "base_url": "http://127.0.0.1:9999/v1", "model": "offline",
          "semantic_sort": False, "l0_days": 30, "recovery_days": 7}


@pytest.fixture
def bank(tmp_path):
    store = Store(tmp_path / "bank")
    configure_review(store, True)
    return store


def evidence(store, session="a"):
    store.task_start(session, "Please check the backup", now=10)
    store.capture(session, [{"role": "tool", "content": json.dumps(
        {"success": True, "message": "Backup checksum matched"})}], now=11)


def draft(job, *_args, **_kwargs):
    return {"records": [{"layer": "L1", "kind": "success", "label": "Backup check",
                         "content": "The backup tool reported a matching checksum.",
                         "evidence_ids": [job["input"]["sources"][0]["id"]]}]}


def conflict(action="keep", content="", refs=None):
    return {"action": action, "content": content, "conflicts": refs or [], "reason": "Owner should inspect."}


def pending(store, kind):
    return next(p for p in store.snapshot()["memory_proposals"].values()
                if p["status"] == "pending" and p["job"]["kind"] == kind)


def add(store, content="Old knowledge", layer="L1", persistent=False):
    store.open_checkpoint("manual", [{"id": "p", "kind": "success", "text": "Owner checked"}], now=10)
    key = store.append("manual", layer=layer, kind="success", label="Test", content=content,
                       evidence_ids=["p"], now=11)
    if persistent:
        with store.transaction() as state:
            next(r for r in state["records"] if r["id"] == key)["bucket"] = "persistent"
    return key


def test_generation_needs_owner_then_current_conflict_acceptance(bank):
    evidence(bank)
    assert run_review(bank, CONFIG, request=draft, now=400)["status"] == "proposed"
    assert bank.snapshot()["records"] == []
    assert bank.checkpoint("a", now=400) is None
    first = pending(bank, "generate")
    # Owner decisions use the normal CLI dispatch, not a model bridge.
    COMMANDS["accept-memory"](bank, SimpleNamespace(id=first["id"]))
    assert bank.snapshot()["records"][0]["bucket"] == "temporary"
    assert bank.consolidate(now=20000) == 0
    assert run_review(bank, CONFIG, request=lambda *_: conflict(), now=401)["status"] == "proposed"
    decide(bank, pending(bank, "conflict")["id"], True, now=402)
    assert bank.consolidate(now=20000) == 1


@pytest.mark.parametrize("action", ["keep", "rewrite", "discard"])
def test_conflict_decision_only_modifies_temp_after_manual_acceptance(bank, action):
    old = add(bank, persistent=True)
    target = add(bank, "New conflicting knowledge")
    before = copy.deepcopy(bank.snapshot()["records"])
    def reviewer(job, *_):
        assert job["kind"] == "conflict"
        assert job["input"]["temporary"]["id"] == target
        assert [r["id"] for r in job["input"]["persistent"]] == [old]
        return conflict(action, "Rewritten temporary lesson" if action == "rewrite" else "", [old])
    run_review(bank, CONFIG, request=reviewer, now=400)
    assert bank.snapshot()["records"] == before
    proposal = pending(bank, "conflict")
    decide(bank, proposal["id"], True, now=401)
    records = bank.snapshot()["records"]
    assert next(r for r in records if r["id"] == old) == before[0]
    if action == "discard":
        assert not any(r["id"] == target for r in records)
        assert proposal["job"]["input"]["temporary"]["content"] == "New conflicting knowledge"
    else:
        expected = "Rewritten temporary lesson" if action == "rewrite" else "New conflicting knowledge"
        assert next(r for r in records if r["id"] == target)["content"] == expected
        assert bank.consolidate(now=20000) == 1
    with pytest.raises(ValueError):
        decide(bank, proposal["id"], True, now=402)


def test_rejecting_preserves_original_and_never_promotes(bank):
    key = add(bank)
    run_review(bank, CONFIG, request=lambda *_: conflict("discard"), now=400)
    decide(bank, pending(bank, "conflict")["id"], False, now=401)
    assert bank.snapshot()["records"][0]["id"] == key
    assert bank.consolidate(now=20000) == 0


def test_same_layer_only_and_changed_persistent_invalidates_approval(bank):
    old = add(bank, persistent=True)
    add(bank, layer="L2", persistent=True)
    add(bank, "New")
    def reviewer(job, *_):
        assert {r["layer"] for r in job["input"]["persistent"]} == {"L1"}
        return conflict()
    run_review(bank, CONFIG, request=reviewer, now=400)
    decide(bank, pending(bank, "conflict")["id"], True, now=401)
    bank.owner_edit(old, content="Owner corrected persistent knowledge")
    assert bank.consolidate(now=20000) == 0
    assert run_review(bank, CONFIG, request=reviewer, now=402)["status"] == "proposed"


def test_stale_decision_and_sort_bypass_are_refused(bank):
    old = add(bank, persistent=True)
    key = add(bank, "New")
    run_review(bank, CONFIG, request=lambda *_: conflict(), now=400)
    proposal = pending(bank, "conflict")
    bank.owner_edit(old, content="Changed")
    with pytest.raises(PermissionError):
        decide(bank, proposal["id"], True, now=401)
    state = bank.snapshot()
    with pytest.raises(PermissionError):
        bank.apply_sort_plan(state["revision"], [{"id": key, "layer": "L2"}], now=20000)
    configure_review(bank, False)
    assert bank.consolidate(now=20000) == 0  # disabling cannot bypass already gated records


@pytest.mark.parametrize("mutation", ["foreground", "finished_foreground", "source_edit"])
def test_inflight_results_discarded_when_activity_or_source_changes(bank, mutation):
    add(bank)
    def reviewer(*_):
        if mutation == "source_edit":
            bank.owner_edit(bank.snapshot()["records"][0]["id"], content="Changed")
        else:
            bank.activity("foreground", True, now=401)
            if mutation == "finished_foreground":
                bank.activity("foreground", False, now=402)
        return conflict()
    assert run_review(bank, CONFIG, request=reviewer, now=400)["status"] == "stale_or_busy"
    assert bank.snapshot().get("memory_proposals", {}) == {}


def test_failed_request_not_automatically_repeated_and_source_survives(bank):
    add(bank)
    calls = []
    def broken(*_):
        calls.append(True)
        raise OSError("offline")
    assert run_review(bank, CONFIG, request=broken, now=400)["status"] == "failed"
    assert run_review(bank, CONFIG, request=broken, now=401)["status"] == "nothing_pending"
    assert len(calls) == 1
    retry_failed(bank)
    assert run_review(bank, CONFIG, request=lambda *_: conflict(), now=402)["status"] == "proposed"
    assert bank.snapshot()["records"][0]["bucket"] == "temporary"


@pytest.mark.parametrize("bad", [
    {"action": "keep", "content": "", "conflicts": ["other-layer-id"], "reason": "bad"},
    {"action": "keep", "content": "illegal rewrite", "conflicts": [], "reason": "bad"},
    {"action": "delete_persistent", "content": "", "conflicts": [], "reason": "bad"},
    {"action": "rewrite", "content": "password=not-a-real-password", "conflicts": [], "reason": "bad"},
])
def test_invalid_conflict_output_cannot_change_memory(bank, bad):
    add(bank)
    before = bank.snapshot()["records"]
    assert run_review(bank, CONFIG, request=lambda *_: bad, now=400)["status"] == "failed"
    assert bank.snapshot()["records"] == before


def test_full_layer_over_budget_refuses_without_calling_model(bank):
    for index in range(33):
        add(bank, str(index), persistent=True)
    add(bank, "Pending")
    def forbidden(*_):
        raise AssertionError("Oversized layer must not be sent partially")
    assert run_review(bank, CONFIG, request=forbidden, now=400)["status"] == "failed"
    assert bank.consolidate(now=20000) == 0


def test_busy_recent_idle_and_duplicate_generation_do_not_infer(bank):
    evidence(bank)
    assert run_review(bank, CONFIG, request=draft, now=200)["status"] == "not_idle"
    bank.activity("busy", True, now=201)
    assert run_review(bank, CONFIG, request=draft, now=900)["status"] == "not_idle"
    bank.activity("busy", False, now=901)
    run_review(bank, CONFIG, request=draft, now=1300)
    assert run_review(bank, CONFIG, request=draft, now=1301)["status"] == "nothing_pending"


def test_acceptance_rechecks_generation_sources(bank):
    evidence(bank)
    run_review(bank, CONFIG, request=draft, now=400)
    proposal = pending(bank, "generate")
    bank.task_start("a", "Later correction", now=401)
    with pytest.raises(PermissionError):
        decide(bank, proposal["id"], True, now=402)
    assert bank.snapshot()["records"] == []


def test_model_surface_cannot_accept_or_discard(bank):
    surface = Surface(bank, "a")
    with pytest.raises(ValueError):
        surface.dispatch({"action": "accept-memory", "id": "x"})
    surface.dispatch({"action": "catalog", "section": "checkpoint"})
    with pytest.raises(ValueError):
        surface.dispatch({"action": "describe", "section": "checkpoint", "name": "accept-memory"})


def test_owner_parser_and_service_owner_commands_share_decision_path(bank):
    parser = argparse.ArgumentParser()
    register_cli(parser)
    args = parser.parse_args(["reject-memory", "proposal-id"])
    assert args.id == "proposal-id"
    assert args.memory_command == "reject-memory"
    from local_extensions.e4b_memory.service import OWNER_COMMANDS
    assert OWNER_COMMANDS["accept-memory"] is COMMANDS["accept-memory"]


def test_local_maintenance_consumes_review_mode_and_not_semantic_sort(bank, monkeypatch):
    evidence(bank)
    provider = E4BMemoryProvider()
    provider.store, provider.settings = bank, CONFIG
    from local_extensions.e4b_memory import review
    calls = []
    monkeypatch.setattr(review, "run_review", lambda *_: calls.append(True))
    provider.maintain()
    assert calls == [True]


def test_settings_opt_in_requires_explicit_local_main_model(monkeypatch):
    from hermes_cli import config
    value = {"memory": {"e4b_memory": {"automatic_review": True}},
             "model": {"provider": "custom", "default": "offline", "base_url": CONFIG["base_url"]}}
    monkeypatch.setattr(config, "load_config_readonly", lambda: value)
    assert load_settings()["automatic_review"] is True
    value["model"]["base_url"] = "https://example.com"
    with pytest.raises(ValueError):
        load_settings()
    service_config = {k: v for k, v in CONFIG.items() if k != "automatic_review"}
    assert settings(service_config) == service_config  # old settings stay valid
    with pytest.raises(ValueError):
        settings({**service_config, "automatic_review": "yes"})


def test_service_credentials_cannot_decide_proposals(tmp_path):
    runtime = create(tmp_path / "service")
    tokens = runtime.credentials.issue()
    body = {"session": "a", "instance": "one", "action": "begin", "arguments": {"message": "Task"}}
    runtime.dispatch(tokens["controller"], body)
    for role in ("controller", "model"):
        with pytest.raises(PermissionError):
            runtime.dispatch(tokens[role], {**body, "action": "accept-memory", "arguments": {"id": "x"}})


def test_http_review_has_no_tools_and_rejects_duplicate_json(monkeypatch):
    from local_extensions.e4b_memory import review_model
    sent = []
    class Transport:
        def open(self, request, timeout):
            sent.append(json.loads(request.data))
            assert timeout == 120
            result = {"choices": [{"finish_reason": "stop", "message": {
                "content": '{"records":[],"records":[]}'}}]}
            return io.BytesIO(json.dumps(result).encode())
    monkeypatch.setattr(review_model, "build_opener", lambda *_: Transport())
    with pytest.raises(ValueError):
        review_model.request_review({"kind": "generate", "input": {}}, CONFIG["base_url"], "offline")
    assert "tools" not in sent[0]


def test_new_restricted_profile_exposes_review_opt_in_without_activation(tmp_path):
    from local_extensions.e4b_memory.restricted import create_profile
    profile = tmp_path / "new-profile"
    create_profile(profile, CONFIG["base_url"], "offline", automatic_review=True)
    config = json.loads((profile / "config.yaml").read_text())
    assert config["memory"]["e4b_memory"]["automatic_review"] is True
    assert config["platform_toolsets"]["cli"] == ["memory"]
    assert not (profile / "e4b_memory" / "memory.json").exists()


def test_service_worker_runs_one_scripted_review_and_owner_can_inspect(tmp_path, monkeypatch):
    runtime = create(tmp_path / "service")
    configure_review(runtime.store, True)
    evidence(runtime.store)
    from local_extensions.e4b_memory import review_model
    monkeypatch.setattr(review_model, "request_review", draft)
    runtime._review_tick(CONFIG)
    assert pending(runtime.store, "generate")["verified"] is False
    assert runtime.store.snapshot()["records"] == []


def test_multiple_same_layer_records_cannot_use_one_stale_promotion_snapshot(bank):
    add(bank, "First")
    add(bank, "Second")
    for now in (400, 402):
        run_review(bank, CONFIG, request=lambda *_: conflict(), now=now)
        decide(bank, pending(bank, "conflict")["id"], True, now=now + 1)
    assert bank.consolidate(now=20000) == 1
    assert sum(r["bucket"] == "temporary" for r in bank.snapshot()["records"]) == 1


def test_readback_failure_does_not_roll_back_or_repeat_accepted_write(bank, monkeypatch):
    evidence(bank)
    run_review(bank, CONFIG, request=draft, now=400)
    proposal = pending(bank, "generate")
    original = bank.snapshot
    monkeypatch.setattr(bank, "snapshot", lambda: (_ for _ in ()).throw(OSError("readback")))
    with pytest.raises(OSError):
        decide(bank, proposal["id"], True, now=401)
    monkeypatch.setattr(bank, "snapshot", original)
    assert len(bank.snapshot()["records"]) == 1
    with pytest.raises(ValueError):
        decide(bank, proposal["id"], True, now=402)


@pytest.mark.parametrize("change", ["missing_source", "invented_preference", "extra_authority"])
def test_generation_cannot_invent_source_ids_or_approval(bank, change):
    evidence(bank)
    def invalid(job, *_):
        result = draft(job)
        record = result["records"][0]
        if change == "missing_source":
            record["evidence_ids"] = ["invented"]
        elif change == "invented_preference":
            record.update(layer="L3", kind="preference")
        else:
            record["approved"] = True
        return result
    assert run_review(bank, CONFIG, request=invalid, now=400)["status"] == "failed"
    assert bank.snapshot()["records"] == []


def test_disk_failure_before_owner_commit_preserves_pending_draft(bank, monkeypatch):
    evidence(bank)
    run_review(bank, CONFIG, request=draft, now=400)
    proposal = pending(bank, "generate")
    def fail_write(_state):
        raise OSError("disk unavailable")
    monkeypatch.setattr(bank, "_write", fail_write)
    with pytest.raises(OSError):
        decide(bank, proposal["id"], True, now=401)
    assert bank.snapshot()["records"] == []
    assert pending(bank, "generate")["id"] == proposal["id"]


def test_service_foreground_request_does_not_wait_for_background_review(tmp_path, monkeypatch):
    import threading
    import time
    from local_extensions.e4b_memory import review_model
    runtime = create(tmp_path / "service")
    config_path = runtime.root / "settings.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
    evidence(runtime.store)
    # Keep fixture evidence within retention while leaving it idle.
    with runtime.store.transaction() as state:
        state["tasks"]["a"]["updated"] = time.time()
    started, release = threading.Event(), threading.Event()
    def slow(job, *_args, **_kwargs):
        started.set()
        assert release.wait(5), "test must release scripted inference"
        return draft(job)
    monkeypatch.setattr(review_model, "request_review", slow)
    try:
        runtime.maintain()
        assert started.wait(5)
        tokens = runtime.credentials.issue()
        body = {"session": "live", "instance": "one", "action": "begin", "arguments": {"message": "New work"}}
        assert runtime.dispatch(tokens["controller"], body)["started"] is True
    finally:
        release.set()
        if runtime.review_worker:
            runtime.review_worker.join(timeout=5)
    assert not runtime.review_worker.is_alive()
    assert runtime.store.snapshot().get("memory_proposals", {}) == {}


@pytest.mark.parametrize("response", [
    {"choices": None}, {"choices": "x"}, {"choices": [None]},
    {"choices": [{"message": []}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": '{"records":[]}', "tool_calls": [{"id": "x"}]}}]},
    {"choices": [{"finish_reason": "length", "message": {"content": '{"records":[]}'}}]},
])
def test_malformed_incomplete_or_tool_bearing_responses_are_refused(monkeypatch, response):
    from local_extensions.e4b_memory import review_model
    class Transport:
        def open(self, *_args, **_kwargs):
            return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(review_model, "build_opener", lambda *_: Transport())
    with pytest.raises(ValueError):
        review_model.request_review({"kind": "generate", "input": {}}, CONFIG["base_url"], "offline")
