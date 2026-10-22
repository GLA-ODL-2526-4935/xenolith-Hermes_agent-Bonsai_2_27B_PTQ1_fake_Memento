"""Offline checkpoint-v2 candidate and shadow capture, NOT a compression provider."""
import copy
import time

from .step_policy import finite_timestamp
from .step_recovery import digest
from .store import text
from .transcript import source_events, visible_transcript


def capture_shadow(store, session, messages, *, now=None):
    """Save source steps, then commit a manifest only after every receipt reads back.

    A crash can leave unreferenced steps. Retrying the same transcript reconciles them
    idempotently; no tool is replayed and no successful manifest is invented.
    """
    text(session, 200)
    messages, pending = visible_transcript(messages)
    events = source_events(messages)
    if not events:
        raise ValueError("No visible user/tool source to checkpoint")
    now = finite_timestamp(time.time() if now is None else now)
    signature = digest(messages)
    old = read_checkpoint(store, session)
    if old:
        prefix = old["messages"]
        if messages[:len(prefix)] != prefix:
            raise PermissionError("Transcript was rewound/rebuilt; owner reconciliation required")
        if old["transcript_digest"] == signature:
            return receipt(old)
    references = []
    for event in events:
        payload = {"step_id": "shadow-" + event["id"],
                   "expected_revision": store.step_recovery(session)["revision"],
                   "status": "in_progress", "events": [event], "facts": [
                       {"category": "requirement" if event["role"] == "user" else "observation",
                        "event_id": event["id"], "quote": event["content"]}]}
        saved = store.save_step(session, payload, now=now)
        references.append({key: saved[key] for key in ("step_id", "revision", "digest")})
    record = {"version": 2, "revision": (old["revision"] if old else 0) + 1,
              "saved": now, "transcript_digest": signature, "messages": messages,
              "references": references, "pending_tool_calls": pending}
    record["integrity"] = digest(record)
    with store.transaction() as state:
        current = state.setdefault("shadow_checkpoints", {}).get(session)
        if current != old:
            raise PermissionError("Concurrent checkpoint changed; reconcile before retry")
        state["shadow_checkpoints"][session] = copy.deepcopy(record)
    saved = read_checkpoint(store, session)
    if saved != record:
        raise OSError("Checkpoint readback failed; retain original context")
    return receipt(saved)


def read_checkpoint(store, session):
    record = store.snapshot().get("shadow_checkpoints", {}).get(session)
    if record is None:
        return None
    keys = {"version", "revision", "saved", "transcript_digest", "messages", "references",
            "pending_tool_calls", "integrity"}
    if (not isinstance(record, dict) or set(record) != keys or record["version"] != 2
            or type(record["revision"]) is not int or record["revision"] < 1):
        raise ValueError("Invalid checkpoint structure")
    finite_timestamp(record["saved"])
    if digest({k: v for k, v in record.items() if k != "integrity"}) != record["integrity"]:
        raise ValueError("Checkpoint integrity mismatch")
    messages, pending = visible_transcript(record["messages"])
    if digest(messages) != record["transcript_digest"] or pending != record["pending_tool_calls"]:
        raise ValueError("Checkpoint transcript mismatch")
    events = source_events(messages)
    if not isinstance(record["references"], list) or len(record["references"]) != len(events):
        raise ValueError("Checkpoint coverage mismatch")
    for event, ref in zip(events, record["references"]):
        step = store.step_recovery(session, "shadow-" + event["id"])
        if (ref != {k: step[k] for k in ("step_id", "revision", "digest")}
                or step["events"] != [event]):
            raise ValueError("Checkpoint evidence mismatch")
    return copy.deepcopy(record)


def receipt(record):
    return {"checkpoint_version": 2, "revision": record["revision"],
            "transcript_digest": record["transcript_digest"], "readback": True,
            "coverage": "accepted_visible_text_only", "shadow_only": True,
            "context_replacement_allowed": False}


def require_checkpoint(store, session, messages):
    """Candidate gate for a FUTURE consumer; never grants permission to compress."""
    messages, pending = visible_transcript(messages)
    record = read_checkpoint(store, session)
    if not record or record["transcript_digest"] != digest(messages):
        raise PermissionError("Current transcript has no matching durable checkpoint")
    if pending:
        raise PermissionError("Pending tool calls must remain in the live transcript")
    return receipt(record)


def preview_rebuild(store, session):
    record = read_checkpoint(store, session)
    if record is None:
        raise ValueError("No checkpoint for this session")
    from .task_state import task_view
    # Lossless baseline: no assistant prose, schema, evidence or tool envelope is dropped.
    # Shortening needs model acceptance; a preview must not pretend this saves tokens.
    return {"mode": "lossless_preview", "messages": record["messages"],
            "task_state": task_view(store, session), "receipt": receipt(record),
            "pending_tool_calls": record["pending_tool_calls"], "shortening_enabled": False,
            "token_savings": None, "live_context_changed": False}


def try_shadow(store, session, messages):
    """The async observation hook cannot suppress normal turn completion on refusal."""
    try:
        return capture_shadow(store, session, messages)
    except (OSError, ValueError, PermissionError, TypeError, KeyError):
        return {"shadow_only": True, "readback": False, "context_replacement_allowed": False,
                "error": "Shadow capture refused or unavailable; original context must be retained"}
