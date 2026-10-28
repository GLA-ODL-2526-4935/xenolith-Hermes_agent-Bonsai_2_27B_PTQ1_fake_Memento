"""Owner-accepted proposals, never autonomous memory writes or verification."""
import copy
import json
import time
import uuid

from .step_recovery import digest
from .step_policy import reject_sensitive_text


def configure_review(store, enabled):
    if type(enabled) is not bool:
        raise ValueError("automatic_review must be boolean")
    with store.transaction() as state:
        if state.get("automatic_review", False) != enabled:
            state["automatic_review"] = enabled
        if enabled:
            for record in state["records"]:
                if record["bucket"] == "temporary":
                    record["review_required"] = True


def persistent_layer(state, layer):
    return [copy.deepcopy(r) for r in state["records"]
            if r["bucket"] == "persistent" and r["layer"] == layer]


def conflict_basis(state, record):
    return digest({"record": {k: record[k] for k in ("id", "layer", "kind", "label", "content", "evidence")},
                   "persistent": persistent_layer(state, record["layer"])})


def can_promote(state, record):
    if not state.get("automatic_review") and not record.get("review_required"):
        return True
    return record.get("conflict_accepted") == conflict_basis(state, record)


def generation_input(state, session):
    task = state["tasks"].get(session, {})
    sources = copy.deepcopy(task.get("evidence", []))
    if not sources:
        return None
    if task.get("goal"):
        sources.append({"id": "user-goal", "kind": "user_excerpt", "text": task["goal"]})
    return {"session": session, "sources": sources}


def next_job(state):
    proposals = state.get("memory_proposals", {})
    if len(proposals) >= 128:
        raise ValueError("Proposal capacity reached; owner export/cleanup required")
    attempted = state.get("memory_review_attempts", {})
    attempted = {**state.get("memory_review_archived_bases", {}), **attempted}
    # Resolve existing temporary records first; never mix layers in one review.
    for record in state["records"]:
        if record["bucket"] != "temporary" or can_promote(state, record):
            continue
        value = {"temporary": copy.deepcopy(record), "persistent": persistent_layer(state, record["layer"])}
        basis = conflict_basis(state, record)
        if basis not in attempted:
            return {"kind": "conflict", "target": record["id"], "basis": basis, "input": value}
    for session in state["tasks"]:
        value = generation_input(state, session)
        if value:
            basis = digest(value)
            if basis not in attempted:
                return {"kind": "generate", "target": session, "basis": basis, "input": value}
    return None


def validate_result(job, result):
    from .store import validate_record, text
    if not isinstance(result, dict):
        raise ValueError("Review result must be an object")
    if job["kind"] == "generate":
        if set(result) != {"records"} or not isinstance(result["records"], list) or len(result["records"]) > 4:
            raise ValueError("Expected at most four memory drafts")
        sources = {s["id"]: s for s in job["input"]["sources"]}
        for record in result["records"]:
            if not isinstance(record, dict) or set(record) != {"layer", "kind", "label", "content", "evidence_ids"}:
                raise ValueError("Invalid draft fields")
            validate_record({**record, "bucket": "temporary"})
            refs = record["evidence_ids"]
            if (not isinstance(refs, list) or not 1 <= len(refs) <= 5
                    or any(not isinstance(ref, str) or ref not in sources for ref in refs)):
                raise ValueError("Draft needs supplied source IDs")
            if record["kind"] in {"preference", "decision"} and "user-goal" not in refs:
                raise ValueError("User preference/decision needs a user source")
            if record["kind"] == "critical_failure" and not any(sources[r]["kind"] == "critical_failure" for r in refs):
                raise ValueError("Critical failure needs reported critical evidence")
    else:
        if set(result) != {"action", "content", "conflicts", "reason"}:
            raise ValueError("Invalid conflict fields")
        if result["action"] not in {"keep", "rewrite", "discard"}:
            raise ValueError("Invalid conflict decision")
        text(result["reason"], 1200)
        refs = result["conflicts"]
        ids = {r["id"] for r in job["input"]["persistent"]}
        if not isinstance(refs, list) or any(not isinstance(r, str) or r not in ids for r in refs):
            raise ValueError("Conflict references must belong to the compared layer")
        if result["action"] == "rewrite":
            text(result["content"])
        elif result["content"] != "":
            raise ValueError("Only rewrite may provide replacement text")
    reject_sensitive_text(result)


def run_review(store, settings, *, request=None, now=None):
    """At most one request per maintenance tick; no lock held during inference."""
    from .review_model import request_review
    clock = (lambda: time.time()) if now is None else (lambda: now)
    state = store.snapshot()
    if not state.get("automatic_review") or state["active"] or clock() - state["last_activity"] < 300:
        return {"status": "not_idle"}
    job = next_job(state)
    if job is None:
        return {"status": "nothing_pending"}
    result, error = None, None
    try:
        # Entire same-layer coverage or refusal, never pretend a partial search is complete.
        if len(job["input"].get("persistent", [])) > 32 or len(json.dumps(job["input"]).encode()) > 18000:
            raise ValueError("Review scope exceeds budget")
        reject_sensitive_text(job["input"])
        if request:
            result = request(job, settings["base_url"], settings["model"])
        else:
            result = request_review(job, settings["base_url"], settings["model"],
                                    timeout=settings.get("review_timeout_seconds", 120))
        validate_result(job, result)
    except (OSError, ValueError, TypeError, KeyError, PermissionError):
        error = "Review failed or exceeded budget; sources retained. Owner may retry."
    with store.transaction() as current:
        if current["revision"] != state["revision"] or current["active"] or clock() - current["last_activity"] < 300:
            return {"status": "stale_or_busy"}
        attempts = current.setdefault("memory_review_attempts", {})
        if len(attempts) >= 256:
            raise ValueError("Review attempt capacity reached; owner cleanup required")
        attempts[job["basis"]] = "failed" if error else "proposed"
        if error:
            current["memory_review_error"] = error
        else:
            key = uuid.uuid4().hex
            current.setdefault("memory_proposals", {})[key] = {
                "id": key, "job": job, "result": copy.deepcopy(result), "status": "pending",
                "created": clock(), "verified": False, "grants_permission": False}
    if not error and store.snapshot().get("memory_proposals", {}).get(key, {}).get("job") != job:
        raise OSError("Proposal readback failed; original memory retained")
    return {"status": "failed" if error else "proposed"}


def decide(store, proposal_id, accept, *, now=None, expected_digest=None, selected_drafts=None):
    """OWNER ONLY. Accept exact inspected content, or reject without changing records."""
    from .store import fingerprint
    now = time.time() if now is None else now
    with store.transaction() as state:
        if state["active"]:
            raise PermissionError("Wait for foreground work to finish")
        proposal = state.get("memory_proposals", {}).get(proposal_id)
        if not proposal or proposal["status"] != "pending":
            raise ValueError("No pending proposal with that ID")
        if expected_digest is not None and digest(proposal) != expected_digest:
            raise PermissionError("Proposal changed since inspection; inspect again")
        job, result = proposal["job"], proposal["result"]
        validate_result(job, result)
        selection = _selection(job, result, accept, selected_drafts)
        if accept:
            chosen = result if selection is None else {"records": [result["records"][i - 1] for i in selection]}
            _accept(state, job, chosen, now, fingerprint)
        status = "accepted" if accept else "rejected"
        if selection is not None:
            proposal["accepted_drafts"] = selection
            proposal["rejected_drafts"] = [i for i in range(1, len(result["records"]) + 1) if i not in selection]
            if proposal["rejected_drafts"]:
                status = "partially_accepted"
        proposal.update(status=status, decided=now)
    saved = store.snapshot()["memory_proposals"][proposal_id]
    if saved["status"] != status:
        raise OSError("Decision readback failed; reconcile, do not repeat side effects")
    return {"id": proposal_id, "status": saved["status"]}


def _selection(job, result, accept, selected):
    if selected is None:
        return None
    if (not accept or job["kind"] != "generate" or not isinstance(selected, list) or not selected
            or any(type(i) is not int or not 1 <= i <= len(result["records"]) for i in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Select distinct one-based draft numbers on generation acceptance only")
    return sorted(selected)


def _accept(state, job, result, now, fingerprint):
    if job["kind"] == "generate":
        if digest(generation_input(state, job["target"])) != job["basis"]:
            raise PermissionError("Draft source changed; reject and regenerate")
        sources = {s["id"]: s for s in job["input"]["sources"]}
        for draft in result["records"]:
            record = {k: draft[k] for k in ("layer", "kind", "label", "content")}
            record.update(id=uuid.uuid4().hex, bucket="temporary", created=now, session=job["target"],
                          evidence=[copy.deepcopy(sources[r]) for r in dict.fromkeys(draft["evidence_ids"])],
                          review_required=True, owner_accepted=now, proposal_basis=job["basis"])
            if not any(fingerprint(r) == fingerprint(record) for r in state["records"]):
                state["records"].append(record)
    else:
        record = next((r for r in state["records"] if r["id"] == job["target"] and r["bucket"] == "temporary"), None)
        if record is None or conflict_basis(state, record) != job["basis"]:
            raise PermissionError("Temporary or persistent source changed; re-review required")
        if result["action"] == "discard":
            state["records"].remove(record)
        else:
            if result["action"] == "rewrite":
                record["content"] = result["content"]
                record["owner_edited"] = now
            record["conflict_accepted"] = conflict_basis(state, record)
            record["review_required"] = True


def retry_failed(store):
    """OWNER ONLY: allow a deliberate retry, never an automatic inference loop."""
    with store.transaction() as state:
        attempts = state.get("memory_review_attempts", {})
        state["memory_review_attempts"] = {k: v for k, v in attempts.items() if v != "failed"}
        state.pop("memory_review_error", None)
    return {"retry_enabled": True}
