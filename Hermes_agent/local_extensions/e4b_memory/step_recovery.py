"""Controller-submitted visible step evidence; never verified or curated memory."""
import copy
import hashlib
import json
import time

from .step_policy import check_fact_source, finite_timestamp, reference_view, reject_sensitive_text

MAX_STEP_BYTES = 12000
MAX_SESSION_STEPS = 32
CATEGORIES = {"requirement", "decision", "observation", "uncertainty", "next_action", "rejected"}
STATUSES = {"proposed", "in_progress", "blocked", "completed", "failed", "cancelled"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_step(payload):
    """Quotes establish provenance, not factual truth, completeness or authority."""
    from .store import text
    fields = {"step_id", "expected_revision", "status", "events", "facts"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Expected step_id, expected_revision, status, events and facts only")
    step_id = text(payload["step_id"], 100)
    if step_id != payload["step_id"]:
        raise ValueError("Step ID cannot contain surrounding whitespace")
    revision = payload["expected_revision"]
    if type(revision) is not int or revision < 0:
        raise ValueError("Expected a nonnegative integer revision")
    if not isinstance(payload["status"], str) or payload["status"] not in STATUSES:
        raise ValueError("Unknown controller-reported step status")
    events = _events(payload["events"])
    facts = payload["facts"]
    if not isinstance(facts, list) or not 1 <= len(facts) <= 32:
        raise ValueError("Expected 1–32 source-linked facts")
    for fact in facts:
        if not isinstance(fact, dict) or set(fact) != {"category", "event_id", "quote"}:
            raise ValueError("Facts require category, event_id and quote only")
        category = text(fact["category"], 30)
        event_id = text(fact["event_id"], 100)
        text(fact["quote"], 4000)
        if (category != fact["category"] or event_id != fact["event_id"] or category not in CATEGORIES
                or event_id not in events or fact["quote"] not in events[event_id]["content"]):
            raise ValueError("Fact must quote an included source event exactly")
        check_fact_source(category, fact["quote"], events[event_id])
    record = {"step_id": step_id, "status": payload["status"],
              "events": list(events.values()), "facts": copy.deepcopy(facts)}
    reject_sensitive_text(record)
    if len(json.dumps(record, ensure_ascii=False).encode("utf-8")) > MAX_STEP_BYTES:
        raise ValueError("Step exceeds recovery budget; split at an explicit safe boundary, never truncate")
    return revision, record


def _events(values):
    from .store import text
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise ValueError("Expected 1–32 visible source events")
    events = {}
    for event in values:
        if not isinstance(event, dict) or set(event) != {"id", "role", "content", "tool_call_id", "outcome"}:
            raise ValueError("Events require id, role, content, tool_call_id and outcome only")
        key = text(event["id"], 100)
        if key != event["id"] or key in events:
            raise ValueError("Duplicate source event")
        role = text(event["role"], 20)
        outcome = text(event["outcome"], 30)
        if role != event["role"] or outcome != event["outcome"]:
            raise ValueError("Role/outcome must use canonical values")
        if role not in {"user", "tool"} or outcome not in {"unknown", "reported_success", "reported_failure"}:
            raise ValueError("Only visible user/tool events with reported outcomes are accepted")
        if role == "user":
            if event["tool_call_id"] != "" or outcome != "unknown":
                raise ValueError("User text cannot assert a tool outcome")
        else:
            text(event["tool_call_id"], 200)
        # Validate without stripping or truncating source text: exact values stay recoverable.
        text(event["content"], 4000)
        events[key] = copy.deepcopy(event)
    return events


class StepRecoveryMixin:
    def save_step(self, session, payload, *, now=None):
        """Controller only. Immutable retries, compare-and-swap and atomic readback."""
        from .store import text
        text(session, 200)
        expected, record = validate_step(payload)
        signature = digest(record)
        now = finite_timestamp(time.time() if now is None else now)
        with self.transaction() as state:
            bank = state.setdefault("step_recovery", {}).setdefault(session, {"revision": 0, "steps": []})
            _check_bank(bank)
            old = next((step for step in bank["steps"] if step["step_id"] == record["step_id"]), None)
            if old:
                if old["digest"] != signature:
                    raise PermissionError("Step ID already committed with different content; use a new step ID")
                receipt = _receipt(old)
            else:
                if expected != bank["revision"]:
                    raise PermissionError("Stale recovery revision; reconcile before retrying")
                if len(bank["steps"]) >= MAX_SESSION_STEPS:
                    raise ValueError("Recovery step limit reached; owner maintenance required, no history was dropped")
                bank["revision"] += 1
                record.update(revision=bank["revision"], digest=signature, saved=now,
                              verified=False, status_source="controller_reported")
                record["integrity"] = _integrity(record)
                bank["steps"].append(record)
                bank["updated"] = now
                state["last_activity"] = now
                receipt = _receipt(record)
        # Do not return a successful checkpoint receipt until the committed data reads back.
        with self.transaction() as state:
            bank = state.get("step_recovery", {}).get(session, {})
            saved = next((s for s in bank.get("steps", []) if s["step_id"] == record["step_id"]), None)
            try:
                if saved is None or _receipt(saved) != receipt or _record_digest(saved) != signature:
                    raise ValueError("Recovery readback mismatch")
                _check_record(saved)
            except ValueError as exc:
                raise OSError("Recovery checkpoint readback failed; context must be retained") from exc
        return receipt

    def step_recovery(self, session, step_id=None):
        bank = self.snapshot().get("step_recovery", {}).get(session, {"revision": 0, "steps": []})
        _check_bank(bank)
        if step_id is None:
            return {"revision": bank["revision"], "steps": [
                {**{k: step[k] for k in ("step_id", "revision", "status", "verified", "digest")},
                 "status_source": "controller_reported", "grants_permission": False}
                for step in bank["steps"]]}
        step = next((s for s in bank["steps"] if s["step_id"] == step_id), None)
        if step is None:
            raise ValueError("Recovery step not found in this session")
        if _record_digest(step) != step["digest"]:
            raise ValueError("Recovery step integrity check failed")
        return reference_view(step)


def _receipt(record):
    if record.get("verified") is not False or record.get("status_source") != "controller_reported":
        raise ValueError("Recovery trust metadata is invalid")
    return {"step_id": record["step_id"], "revision": record["revision"],
            "digest": record["digest"], "saved": record["saved"], "verified": False}


def _record_digest(record):
    return digest({key: record[key] for key in ("step_id", "status", "events", "facts")})


def _integrity(record):
    return digest({key: value for key, value in record.items() if key != "integrity"})


def _check_record(record, *, content_policy=True):
    fields = {"step_id", "status", "events", "facts", "revision", "digest", "saved", "verified", "status_source"}
    if not isinstance(record, dict) or set(record) not in (fields, fields | {"integrity"}):
        raise ValueError("Invalid stored recovery structure")
    _receipt(record)
    if type(record["revision"]) is not int or record["revision"] < 1:
        raise ValueError("Invalid stored step revision")
    finite_timestamp(record["saved"])
    if _record_digest(record) != record["digest"]:
        raise ValueError("Recovery step integrity check failed")
    if "integrity" in record and _integrity(record) != record["integrity"]:
        raise ValueError("Recovery metadata integrity check failed")
    if content_policy:
        validate_step({**{key: record[key] for key in ("step_id", "status", "events", "facts")}, "expected_revision": 0})


def _check_bank(bank, *, content_policy=True):
    if not isinstance(bank, dict) or set(bank) - {"revision", "steps", "updated"}:
        raise ValueError("Invalid recovery bank")
    revision, steps = bank.get("revision"), bank.get("steps")
    if type(revision) is not int or revision < 0 or not isinstance(steps, list) or len(steps) > MAX_SESSION_STEPS:
        raise ValueError("Invalid recovery bank revision/steps")
    seen, previous = set(), 0
    for record in steps:
        _check_record(record, content_policy=content_policy)
        if record["step_id"] in seen or not previous < record["revision"] <= revision:
            raise ValueError("Invalid recovery history ordering")
        seen.add(record["step_id"])
        previous = record["revision"]
    if "updated" in bank:
        finite_timestamp(bank["updated"])


def expire_steps(banks, now, retention):
    result = {}
    for session, bank in banks.items():
        # Old content that now fails the capture policy must still be able to expire.
        _check_bank(bank, content_policy=False)
        remaining = [step for step in bank["steps"] if now - step["saved"] < retention]
        if remaining:
            result[session] = {**bank, "steps": remaining}
    return result
