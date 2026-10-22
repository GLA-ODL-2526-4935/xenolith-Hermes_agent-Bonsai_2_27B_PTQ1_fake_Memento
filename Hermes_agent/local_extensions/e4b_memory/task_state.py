"""Source-linked task ledger; semantic completion/supersession is owner-declared."""
import copy
import time

from .step_policy import finite_timestamp, reject_sensitive_text
from .store import text

STATES = {"pending", "in_progress", "completed", "blocked", "cancelled"}


def task_view(store, session):
    index = store.step_recovery(session)
    events = {}
    for item in index["steps"]:
        for event in store.step_recovery(session, item["step_id"])["events"]:
            if event["id"] in events and events[event["id"]] != event:
                raise ValueError("Conflicting source event IDs")
            events[event["id"]] = event
    items = [{"id": event["id"], "kind": "user_clause" if event["role"] == "user" else "tool_report",
              "source": copy.deepcopy(event), "status": "pending", "superseded_by": None,
              "status_source": "unreviewed"} for event in events.values()]
    by_id = {item["id"]: item for item in items}
    changes = store.snapshot().get("task_annotations", {}).get(session, [])
    for change in changes:
        if change["id"] not in by_id:
            continue
        by_id[change["id"]].update(status=change["status"], status_source="owner_declared",
                                    superseded_by=change["superseded_by"], note=change["note"])
    return {"items": items, "annotation_revision": len(changes), "verified": False,
            "semantic_contract_complete": False, "grants_permission": False}


def annotate_task(store, session, payload, *, now=None):
    fields = {"id", "status", "superseded_by", "note", "expected_revision"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Expected id, status, superseded_by, note and expected_revision")
    if payload["status"] not in STATES:
        raise ValueError("Unknown task status")
    revision = payload["expected_revision"]
    if type(revision) is not int or revision < 0:
        raise ValueError("Invalid annotation revision")
    text(payload["note"], 1200)
    reject_sensitive_text(payload)
    view = task_view(store, session)
    by_id = {item["id"]: item for item in view["items"]}
    key, replacement = payload["id"], payload["superseded_by"]
    if key not in by_id:
        raise ValueError("Unknown task source")
    if replacement is not None:
        order = list(by_id)
        if (replacement not in by_id or order.index(replacement) <= order.index(key)
                or by_id[key]["kind"] != "user_clause" or by_id[replacement]["kind"] != "user_clause"):
            raise ValueError("Supersession requires a later complete user source")
    change = {k: copy.deepcopy(payload[k]) for k in fields - {"expected_revision"}}
    change["saved"] = finite_timestamp(time.time() if now is None else now)
    with store.transaction() as state:
        changes = state.setdefault("task_annotations", {}).setdefault(session, [])
        if len(changes) != revision:
            raise PermissionError("Stale task annotation revision")
        if len(changes) >= 64:
            raise ValueError("Task annotation capacity reached")
        changes.append(change)
    return {"annotation_revision": revision + 1, "status_source": "owner_declared", "verified": False}
