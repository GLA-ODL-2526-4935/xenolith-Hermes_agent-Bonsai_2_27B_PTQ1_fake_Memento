"""Offline Markdown, selective approval and durable owner archive contracts."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_extensions.e4b_memory.cli import COMMANDS, register_cli
from local_extensions.e4b_memory.remote_review import RemoteReview
from local_extensions.e4b_memory.review import configure_review, decide, next_job, run_review
from local_extensions.e4b_memory.review_archive import archive, preview_archive, list_archives, review_archived
from local_extensions.e4b_memory.review_display import disposition, history, proposal_markdown, quote
from local_extensions.e4b_memory.store import Store
from local_extensions.e4b_memory.workflow_switch import set_workflow


@pytest.fixture
def bank(tmp_path):
    store = Store(tmp_path / "bank")
    configure_review(store, True)
    store.task_start("s", "Check the research", now=10)
    store.capture("s", [{"role": "tool", "content": json.dumps({"success": True, "message": "Search completed"})}], now=11)

    def drafts(job, *_):
        return {"records": [{"layer": "L1", "kind": "fact", "label": f"Lesson {i}", "content": f"Reported finding {i}",
                             "evidence_ids": [job["input"]["sources"][0]["id"]]} for i in range(3)]}
    assert run_review(store, {"base_url": "http://127.0.0.1:9999/v1", "model": "offline"}, request=drafts, now=400)["status"] == "proposed"
    return store


def proposal(bank):
    return next(iter(bank.snapshot()["memory_proposals"].values()))


def test_markdown_shows_all_drafts_sources_and_audit(bank):
    value = proposal(bank)
    report = proposal_markdown(value, state=bank.snapshot())
    for expected in ("# Memory review", "Draft 1", "Draft 2", "Draft 3", "sources", "Search completed", "Complete audit", "No tool permissions"):
        assert expected in report
    assert value["job"]["basis"] in report
    assert json.dumps(value, ensure_ascii=False, indent=2) in report
    assert bank.snapshot()["records"] == []


def test_quotes_cannot_escape_dynamic_fence():
    malicious = "```\n<script>alert(1)</script>\n`````\n# pretend acceptance"
    block = quote(malicious)
    assert block.startswith("``````text\n")
    assert block.endswith("\n``````\n")
    assert malicious in block


def test_selective_acceptance_is_atomic_and_unselected_are_rejected(bank):
    original = proposal(bank)
    result = decide(bank, original["id"], True, selected_drafts=[3, 1])
    assert result["status"] == "partially_accepted"
    state = bank.snapshot()
    assert [r["content"] for r in state["records"]] == ["Reported finding 0", "Reported finding 2"]
    assert all(r["bucket"] == "temporary" and r["review_required"] for r in state["records"])
    saved = state["memory_proposals"][original["id"]]
    assert saved["accepted_drafts"] == [1, 3] and saved["rejected_drafts"] == [2]
    assert saved["job"] == original["job"] and saved["result"] == original["result"]
    assert bank.consolidate(now=20000) == 0
    with pytest.raises(ValueError):
        decide(bank, original["id"], True, selected_drafts=[2])


@pytest.mark.parametrize("selection", [[], [0], [4], [1, 1], [True], ["1"], "1", [1.0]])
def test_invalid_selection_preserves_queue_and_records(bank, selection):
    before = bank.snapshot()
    with pytest.raises(ValueError):
        decide(bank, proposal(bank)["id"], True, selected_drafts=selection)
    assert bank.snapshot() == before


def test_selection_not_allowed_on_reject_or_conflict(bank):
    key = proposal(bank)["id"]
    with pytest.raises(ValueError):
        decide(bank, key, False, selected_drafts=[1])
    decide(bank, key, True, selected_drafts=[1])
    run_review(bank, {"base_url": "http://127.0.0.1:9999/v1", "model": "offline"}, now=401,
               request=lambda *_: {"action": "keep", "content": "", "conflicts": [], "reason": "No conflict"})
    key = next(p["id"] for p in bank.snapshot()["memory_proposals"].values() if p["status"] == "pending")
    with pytest.raises(ValueError):
        decide(bank, key, True, selected_drafts=[1])


def test_remote_markdown_selection_uses_same_receipt_gates(bank):
    review = RemoteReview(bank, "123", clock=lambda: 100)
    key = proposal(bank)["id"]
    preview = review.prepare(123, f"!memory show {key}", direct=True)
    assert b"# Memory review" in preview["attachment"] and b"Draft 3" in preview["attachment"]
    review.delivered(preview["receipt"], True)
    command = f"!memory accept {key} {preview['receipt'][1]} 2"
    review.prepare(123, command, direct=True)
    assert [r["content"] for r in bank.snapshot()["records"]] == ["Reported finding 1"]
    with pytest.raises(PermissionError):
        review.prepare(123, command, direct=True)


def test_history_computes_stale_without_modifying_sources(bank):
    assert history(bank, "pending")["proposals"]
    bank.task_start("s", "Changed research", now=500)
    before = bank.snapshot()
    assert history(bank, "pending")["proposals"] == []
    assert history(bank, "stale")["proposals"][0]["status"] == "stale"
    assert disposition(before, proposal(bank)) == "stale"
    assert bank.snapshot() == before
    remote = RemoteReview(bank, "123")
    assert proposal(bank)["id"] not in remote.prepare(123, "!memory pending", direct=True)["text"]


def test_markdown_exposes_background_failure_without_discarding_evidence(bank):
    with bank.transaction() as state:
        state["memory_review_error"] = "Scope exceeded; original sources retained"
    report = COMMANDS["review-memory"](bank, SimpleNamespace(id=None, format="markdown", status="all"))
    assert "Background review error" in report and "original sources retained" in report
    assert "Search completed" in report


def test_archive_requires_decision_or_staleness_and_exact_plan(bank):
    key = proposal(bank)["id"]
    with pytest.raises(PermissionError):
        preview_archive(bank, [key])
    decide(bank, key, False)
    before = bank.snapshot()
    plan = preview_archive(bank, [key])
    assert bank.snapshot() == before
    bank.task_start("other", "foreground input", now=501)
    with pytest.raises(PermissionError):
        archive(bank, plan)
    assert key in bank.snapshot()["memory_proposals"]


@pytest.mark.parametrize("decision", ["accepted", "rejected", "partially_accepted", "stale"])
def test_durable_archive_frees_queue_preserves_records_and_prevents_regeneration(bank, decision):
    key = proposal(bank)["id"]
    original = proposal(bank)
    actions = {"accepted": lambda: decide(bank, key, True), "rejected": lambda: decide(bank, key, False),
               "partially_accepted": lambda: decide(bank, key, True, selected_drafts=[2]),
               "stale": lambda: bank.task_start("s", "Changed goal", now=500)}
    actions[decision]()
    before = copy.deepcopy(bank.snapshot()["records"])
    result = archive(bank, preview_archive(bank, [key]))
    assert result["archived"] == 1 and result["memory_records_changed"] is False
    stored = json.loads(Path(result["backup_file"]).read_text(encoding="utf-8"))
    assert stored["proposals"][key]["job"] == original["job"]
    assert stored["proposals"][key]["result"] == original["result"]
    state = bank.snapshot()
    assert key not in state["memory_proposals"] and state["records"] == before
    assert original["job"]["basis"] in state["memory_review_archived_bases"]
    assert original["job"]["basis"] not in state["memory_review_attempts"]
    if decision == "rejected":
        assert next_job(state) is None
    assert history(bank)["live_proposals"] == 0


def test_archive_failed_backup_or_commit_never_loses_live_proposal(bank, monkeypatch):
    from local_extensions.e4b_memory import review_archive
    key = proposal(bank)["id"]
    decide(bank, key, False)
    plan = preview_archive(bank, [key])
    before = bank.snapshot()
    def failed(*_):
        raise OSError("disk failure")
    monkeypatch.setattr(review_archive, "_save_archive", failed)
    with pytest.raises(OSError):
        archive(bank, plan)
    assert bank.snapshot() == before


def test_archive_after_write_failure_leaves_backup_and_original(bank, monkeypatch):
    key = proposal(bank)["id"]
    decide(bank, key, False)
    plan = preview_archive(bank, [key])
    before = bank.snapshot()
    def failed(*_):
        raise OSError("memory commit failed")
    monkeypatch.setattr(bank, "_write", failed)
    with pytest.raises(OSError):
        archive(bank, plan)
    assert bank.snapshot() == before
    files = list((bank.root / "proposal-archives").glob("*.json"))
    assert len(files) == 1 and key in json.loads(files[0].read_text())["proposals"]


def test_archive_plan_cannot_be_forged_to_remove_current_pending(bank):
    value = proposal(bank)
    from local_extensions.e4b_memory.step_recovery import digest
    plan = {"format": "memory-archive-plan-v1", "revision": bank.snapshot()["revision"], "ids": [value["id"]],
            "digest": digest({value["id"]: value}), "count": 1, "effect": "delete"}
    with pytest.raises(PermissionError):
        archive(bank, plan)


def test_archived_audits_remain_readable_but_never_actionable(bank):
    value = proposal(bank)
    decide(bank, value["id"], False)
    archive(bank, preview_archive(bank, [value["id"]]))
    name = list_archives(bank)["files"][0]["file"]
    text = review_archived(bank, name)
    assert "read-only, not actionable" in text and "Draft 1" in text
    assert value["id"] in review_archived(bank, name, format="json")["proposals"]
    with pytest.raises(ValueError):
        decide(bank, value["id"], True)
    with pytest.raises(ValueError):
        review_archived(bank, "../memory.json")


def test_archive_tamper_and_capacity_fail_without_removal(bank):
    value = proposal(bank)
    decide(bank, value["id"], False)
    plan = preview_archive(bank, [value["id"]])
    changed = {**plan, "digest": "0" * 64}
    with pytest.raises(PermissionError):
        archive(bank, changed)
    with bank.transaction() as state:
        state["memory_review_archived_bases"] = {str(i): "rejected" for i in range(4096)}
    plan = preview_archive(bank, [value["id"]])
    with pytest.raises(ValueError):
        archive(bank, plan)
    assert value["id"] in bank.snapshot()["memory_proposals"]


def test_archive_frees_full_proposal_and_attempt_capacity(bank):
    value = proposal(bank)
    decide(bank, value["id"], False)
    with bank.transaction() as state:
        for index in range(1, 128):
            extra = copy.deepcopy(state["memory_proposals"][value["id"]])
            extra["id"] = f"{index:032x}"
            extra["job"]["basis"] = f"basis-{index}"
            state["memory_proposals"][extra["id"]] = extra
            state["memory_review_attempts"][extra["job"]["basis"]] = "proposed"
    with pytest.raises(ValueError, match="capacity"):
        next_job(bank.snapshot())
    result = archive(bank, preview_archive(bank, list(bank.snapshot()["memory_proposals"])))
    assert result["archived"] == 128 and next_job(bank.snapshot()) is None
    assert bank.snapshot()["memory_review_attempts"] == {}


def test_selection_disk_failure_rolls_back_entire_decision(bank, monkeypatch):
    before = bank.snapshot()
    def failed(*_):
        raise OSError("disk failure")
    monkeypatch.setattr(bank, "_write", failed)
    with pytest.raises(OSError):
        decide(bank, proposal(bank)["id"], True, selected_drafts=[1, 3])
    assert bank.snapshot() == before


def test_owner_cli_registers_markdown_selection_and_archive_without_model_tools(bank):
    parser = argparse.ArgumentParser()
    register_cli(parser)
    key = proposal(bank)["id"]
    args = parser.parse_args(["review-memory", "--id", key])
    assert "Draft 1" in COMMANDS[args.memory_command](bank, args)
    args = parser.parse_args(["review-memory", "--id", key, "--format", "json"])
    assert key in COMMANDS[args.memory_command](bank, args)["proposals"]
    args = parser.parse_args(["accept-memory", key, "--drafts", "2"])
    assert COMMANDS[args.memory_command](bank, args)["status"] == "partially_accepted"
    assert COMMANDS["memory-history"](bank, SimpleNamespace(status="partially_accepted"))["proposals"]


def test_local_workflow_switch_keeps_manual_acceptance_and_generated_profile(tmp_path):
    from local_extensions.e4b_memory.restricted import create_profile, profile_config
    root = create_profile(tmp_path / "profile", "http://127.0.0.1:9999/v1", "offline")
    result = set_workflow(root / "config.yaml", True, kind="local")
    assert result["acceptance"] == "manual" and result["restart_required"]
    assert json.loads((root / "config.yaml").read_text()) == profile_config("http://127.0.0.1:9999/v1", "offline", automatic_review=True)
    set_workflow(root / "config.yaml", False, kind="local")
    assert json.loads((root / "config.yaml").read_text()) == profile_config("http://127.0.0.1:9999/v1", "offline")


def test_service_switch_requires_valid_explicit_local_model(tmp_path):
    config = tmp_path / "settings.json"
    original = {"semantic_sort": False, "l0_days": 30, "recovery_days": 7, "base_url": "", "model": ""}
    config.write_text(json.dumps(original))
    with pytest.raises(ValueError):
        set_workflow(config, True, kind="service")
    assert json.loads(config.read_text()) == original
    original.update(base_url="http://127.0.0.1:9999/v1", model="offline")
    config.write_text(json.dumps(original))
    set_workflow(config, True, kind="service")
    assert json.loads(config.read_text())["automatic_review"] is True
