"""Offline owner-DM contracts; no Discord connection or model required."""
import asyncio
import copy
from types import SimpleNamespace

import pytest

from local_extensions.e4b_memory.remote_review import RemoteReview, handle_message
from local_extensions.e4b_memory.review import configure_review, run_review
from local_extensions.e4b_memory.store import Store


@pytest.fixture
def review(tmp_path):
    store = Store(tmp_path / "bank")
    configure_review(store, True)
    store.task_start("s", "Check a backup", now=10)
    store.capture("s", [{"role": "tool", "content": '{"success":true,"message":"Backup reported successful"}'}], now=11)

    def draft(job, *_):
        return {"records": [{"layer": "L1", "kind": "success", "label": "Backup",
                             "content": "A tool reported the backup succeeded.",
                             "evidence_ids": [job["input"]["sources"][0]["id"]]}]}

    assert run_review(store, {"base_url": "http://127.0.0.1:9999/v1", "model": "offline"},
                      request=draft, now=400)["status"] == "proposed"
    return RemoteReview(store, "12345", clock=lambda: 100)


def key(review):
    return next(iter(review.store.snapshot()["memory_proposals"]))


def show(review):
    result = review.prepare(12345, f"!memory show {key(review)}", direct=True)
    review.delivered(result["receipt"], True)
    return result["receipt"][1]


def message(content, user=12345, bot=False, webhook=None):
    return SimpleNamespace(content=content, author=SimpleNamespace(id=user, bot=bot), webhook_id=webhook)


@pytest.mark.parametrize("user,direct,bot,webhook", [
    (999, True, False, False), (12345, False, False, False),
    (12345, True, True, False), (12345, True, False, True)])
def test_unauthorized_cannot_even_read_bank(review, monkeypatch, user, direct, bot, webhook):
    def forbidden():
        pytest.fail("Unauthorized access reached memory")
    monkeypatch.setattr(review.store, "snapshot", forbidden)
    assert review.prepare(user, "!memory pending", direct=direct, bot=bot, webhook=webhook) is None


@pytest.mark.parametrize("owner", ["", "name", "0", "-1", 12345, "1" * 21])
def test_explicit_owner_identity_required(tmp_path, owner):
    with pytest.raises(ValueError):
        RemoteReview(Store(tmp_path), owner)


@pytest.mark.parametrize("accept", [True, False])
def test_remote_decision_complete_preview_then_one_use(review, accept):
    original = review.store.snapshot()
    response = review.prepare(12345, f"!memory show {key(review)}", direct=True)
    assert b"sources" in response["attachment"] and b"evidence_ids" in response["attachment"]
    assert b"UNVERIFIED" in response["attachment"]
    assert review.store.snapshot() == original
    code = response["receipt"][1]
    command = f"!memory {'accept' if accept else 'reject'} {key(review)} {code}"
    with pytest.raises(PermissionError):
        review.prepare(12345, command, direct=True)
    review.delivered(response["receipt"], True)
    review.prepare(12345, command, direct=True)
    state = review.store.snapshot()
    assert state["memory_proposals"][key(review)]["status"] == ("accepted" if accept else "rejected")
    assert len(state["records"]) == int(accept)
    assert review.store.consolidate(now=20000) == 0
    with pytest.raises(PermissionError):
        review.prepare(12345, command, direct=True)


def expire(review, _code):
    review.clock = lambda: 400


def change_proposal(review, _code):
    proposal_id = key(review)
    with review.store.transaction() as state:
        state["memory_proposals"][proposal_id]["result"]["records"][0]["content"] = "Changed draft"


@pytest.mark.parametrize("mutate", [
    expire, change_proposal,
    lambda r, _: r.store.task_start("s", "Changed goal", now=500),
    lambda r, _: r.store.activity("agent", True, now=500),
    lambda r, _: r.receipts.clear(),
    lambda r, code: r.delivered((key(r), code), False),
], ids=["expired", "proposal", "source", "busy", "restart", "failed_delivery"])
def test_stale_or_unconfirmed_acceptance_fails_closed(review, mutate):
    code = show(review)
    mutate(review, code)
    with pytest.raises((PermissionError, ValueError)):
        review.prepare(12345, f"!memory accept {key(review)} {code}", direct=True)
    assert review.store.snapshot()["records"] == []


def test_new_preview_invalidates_old_and_wrong_codes_do_not_accept(review):
    old = show(review)
    new = show(review)
    assert old != new
    for code in (old, "0" * 32, "💥" * 32):
        with pytest.raises(PermissionError):
            review.prepare(12345, f"!memory accept {key(review)} {code}", direct=True)
    review.prepare(12345, f"!memory accept {key(review)} {new}", direct=True)


def test_pending_lists_ids_not_private_sources_and_oversize_never_truncates(review):
    response = review.prepare(12345, "!memory pending", direct=True)
    assert key(review) in response["text"] and "Backup" not in response["text"]
    proposal_id = key(review)
    with review.store.transaction() as state:
        state["memory_proposals"][proposal_id]["extra"] = "x" * 129000
    with pytest.raises(ValueError):
        show(review)
    assert review.receipts == {}


@pytest.mark.parametrize("command", ["!memory accept all", "!memory approve a", "!memory delete anything",
                                     "!memory " + "x" * 201, "!memory show ../../etc/passwd"])
def test_no_generic_dispatch_or_path_access(review, command):
    before = review.store.snapshot()
    with pytest.raises(ValueError):
        review.prepare(12345, command, direct=True)
    assert review.store.snapshot() == before


def test_transport_delivery_required_and_errors_do_not_repeat_side_effects(review):
    sent = []

    async def send(text, attachment):
        sent.append((text, attachment))

    asyncio.run(handle_message(review, message(f"!memory show {key(review)}"), direct=True, send=send))
    code = review.receipts[key(review)]["code"]
    assert sent[0][1] and review.receipts[key(review)]["delivered"]
    asyncio.run(handle_message(review, message(f"!memory accept {key(review)} {code}"), direct=True, send=send))
    assert "accepted" in sent[-1][0]
    assert len(review.store.snapshot()["records"]) == 1
    asyncio.run(handle_message(review, message(f"!memory accept {key(review)} {code}"), direct=True, send=send))
    assert "refused" in sent[-1][0]
    assert len(review.store.snapshot()["records"]) == 1


def test_transport_unauthorized_and_ordinary_chat_silent(review):
    async def forbidden(*_):
        pytest.fail("Unauthorized response")
    for msg, direct in ((message("!memory pending", user=999), True),
                        (message("!memory pending"), False), (message("hello"), True)):
        asyncio.run(handle_message(review, msg, direct=direct, send=forbidden))


def test_failed_preview_send_does_not_activate_receipt(review):
    async def failed(_text, attachment):
        if attachment:
            raise OSError("Delivery failed")
    asyncio.run(handle_message(review, message(f"!memory show {key(review)}"), direct=True, send=failed))
    assert review.receipts == {}


def test_conflict_rewrite_via_remote_preserves_persistent(review):
    code = show(review)
    review.prepare(12345, f"!memory accept {key(review)} {code}", direct=True)
    with review.store.transaction() as state:
        persistent = copy.deepcopy(state["records"][0])
        persistent.update(id="f" * 32, bucket="persistent", content="Old persistent knowledge")
        state["records"].append(persistent)

    def conflict(job, *_):
        return {"action": "rewrite", "content": "Rewritten temporary knowledge",
                "conflicts": [persistent["id"]], "reason": "Scope changed"}
    run_review(review.store, {"base_url": "http://127.0.0.1:9999/v1", "model": "offline"},
               request=conflict, now=401)
    pending_id = next(p["id"] for p in review.store.snapshot()["memory_proposals"].values() if p["status"] == "pending")
    result = review.prepare(12345, f"!memory show {pending_id}", direct=True)
    review.delivered(result["receipt"], True)
    review.prepare(12345, f"!memory accept {pending_id} {result['receipt'][1]}", direct=True)
    records = review.store.snapshot()["records"]
    assert next(r for r in records if r["id"] == persistent["id"]) == persistent
    assert next(r for r in records if r["bucket"] == "temporary")["content"] == "Rewritten temporary knowledge"
