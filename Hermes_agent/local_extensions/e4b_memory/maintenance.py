"""Controller-owned recovery, review requests, sorting and retention."""
import copy
import hashlib
import json
import time

DAY = 86400


def plain_text(value, maximum=1200):
    if not isinstance(value, str):
        return ""
    from .store import text
    try:
        return text(value[:maximum], maximum)
    except ValueError:
        return "[non-text content omitted]"


def extract_evidence(messages):
    """Tool assertions are REVIEW CANDIDATES, never automatically trusted facts."""
    evidence, error, names = {}, "", {}
    for message in (messages or [])[-200:]:
        for call in message.get("tool_calls") or []:
            names[call.get("id")] = call.get("function", {}).get("name")
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        if message.get("name", names.get(message.get("tool_call_id"))) == "e4b_memory":
            continue
        raw = message["content"]
        if len(raw) > 16000:
            continue
        try:
            result = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(result, dict):
            continue
        kind = _result_kind(result)
        if result.get("success") is False or result.get("error"):
            error = plain_text(str(result.get("error") or "Tool reported failure"), 800)
        if kind:
            content = plain_text(raw)
            key = hashlib.sha256(content.encode()).hexdigest()[:24]
            evidence[key] = {"id": key, "kind": kind, "text": content}
    return list(evidence.values())[-8:], error


def _result_kind(result):
    if set(result) & {"image", "images", "image_url", "image_data", "base64", "audio", "file_bytes", "data"}:
        return None
    if result.get("critical_failure") is True and result.get("success") is False:
        return "critical_failure"
    if result.get("success") is True and not result.get("error"):
        return "success"
    return None


class MaintenanceMixin:
    def task_start(self, session, user_text, now=None):
        now = time.time() if now is None else now
        with self.transaction() as state:
            old = state["tasks"].get(session, {})
            if old.get("status") == "complete":
                old = {}
            state["tasks"][session] = {**old, "goal": plain_text(user_text),
                                       "status": "in_progress", "updated": now}

    def capture(self, session, messages, *, reason=None, now=None):
        now = time.time() if now is None else now
        evidence, error = extract_evidence(messages)
        with self.transaction() as state:
            task = state["tasks"].setdefault(session, {"goal": "", "status": "needs_review"})
            previous = {e["id"]: e for e in task.get("evidence", [])}
            previous.update({e["id"]: e for e in evidence})
            task.update(evidence=list(previous.values())[-8:], updated=now)
            if error:
                task["last_error"] = error
            if reason:
                _request(state, session, reason, now)

    def finish_task_turn(self, session, now=None):
        with self.transaction() as state:
            if session in state["tasks"]:
                state["tasks"][session].update(status="needs_review", updated=time.time() if now is None else now)

    def prepare_idle_checkpoints(self, *, idle_seconds=300, now=None):
        now = time.time() if now is None else now
        with self.transaction() as state:
            if state["active"] or now - state["last_activity"] < idle_seconds:
                return
            for session, task in state["tasks"].items():
                if task.get("status") != "complete":
                    _request(state, session, "idle", now)

    def approve_request(self, session, evidence_ids, now=None):
        """OWNER ONLY. Review selected evidence IDs, then grant the append window."""
        now = time.time() if now is None else now
        with self.transaction() as state:
            request = state["requests"].get(session)
            if not request or now - request["created"] >= DAY:
                raise ValueError("No current checkpoint request")
            by_id = {e["id"]: e for e in request["evidence"]}
            if not evidence_ids or any(key not in by_id for key in evidence_ids):
                raise ValueError("Select evidence IDs from the request")
            approved = {key: {k: by_id[key][k] for k in ("kind", "text")} for key in dict.fromkeys(evidence_ids)}
            state["stages"][session] = {"expires": now + 300, "remaining": 8, "evidence": approved}
            state["tasks"][session]["reviewed_signature"] = request["signature"]
            state["requests"].pop(session)
            state["last_activity"] = now

    def recovery(self, session):
        return copy.deepcopy(self.snapshot()["tasks"].get(session))

    def complete_task(self, session):
        with self.transaction() as state:
            if session not in state["tasks"]:
                raise ValueError("Unknown recovery session")
            state["tasks"][session].update(status="complete", updated=time.time())
            state["requests"].pop(session, None)
            state["stages"].pop(session, None)

    def maintenance_snapshot(self, *, idle_seconds=10800, now=None):
        now = time.time() if now is None else now
        state = self.snapshot()
        if state["active"] or now - state["last_activity"] < idle_seconds:
            return None
        records = [r for r in state["records"] if r["bucket"] == "temporary"]
        return {"revision": state["revision"], "records": records} if records else None

    def apply_sort_plan(self, revision, assignments, *, now=None):
        now = time.time() if now is None else now
        with self.transaction() as state:
            if state["active"] or now - state["last_activity"] < 10800 or state["revision"] != revision:
                raise PermissionError("Sort plan became stale or foreground work resumed")
            temporary = {r["id"]: r for r in state["records"] if r["bucket"] == "temporary"}
            plan = _validate_plan(temporary, assignments)
            for key, layer in plan.items():
                temporary[key].update(layer=layer, bucket="persistent", consolidated=now)
            return len(plan)

    def expire(self, *, now=None, l0_days=30, recovery_days=7):
        if not 1 <= l0_days <= 3650 or not 1 <= recovery_days <= 365:
            raise ValueError("Retention interval out of range")
        now = time.time() if now is None else now
        with self.transaction() as state:
            if state["active"]:
                return 0
            before = len(state["records"])
            state["records"] = [r for r in state["records"] if not (
                r["bucket"] == "persistent" and r["layer"] == "L0"
                and now - r.get("consolidated", r["created"]) >= l0_days * DAY)]
            state["stages"] = {s: g for s, g in state["stages"].items() if g["expires"] > now}
            state["requests"] = {s: r for s, r in state["requests"].items() if now - r["created"] < DAY}
            state["tasks"] = {s: t for s, t in state["tasks"].items() if now - t["updated"] < recovery_days * DAY}
            from .step_recovery import expire_steps
            state["step_recovery"] = expire_steps(state.get("step_recovery", {}), now, recovery_days * DAY)
            # A manifest cannot survive its source steps. Annotations have the same lifetime.
            for section in ("shadow_checkpoints", "task_annotations"):
                state[section] = {s: value for s, value in state.get(section, {}).items()
                                  if s in state["step_recovery"]}
            state["shadow_checkpoints"] = {s: checkpoint for s, checkpoint in state["shadow_checkpoints"].items()
                if all(ref["step_id"] in {step["step_id"] for step in state["step_recovery"][s]["steps"]}
                       for ref in checkpoint["references"])}
            return before - len(state["records"])


def _request(state, session, reason, now):
    task = state["tasks"].get(session, {})
    evidence = task.get("evidence", [])
    if not evidence:
        return
    signature = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    if task.get("reviewed_signature") == signature:
        return
    old = state["requests"].get(session)
    if old and old["signature"] == signature:
        return
    state["requests"][session] = {"reason": reason, "created": now, "signature": signature,
                                   "evidence": copy.deepcopy(evidence), "approved": False}


def _validate_plan(temporary, assignments):
    from .store import LAYERS
    if not isinstance(assignments, list):
        raise ValueError("Sort plan must be a list")
    plan = {}
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != {"id", "layer"}:
            raise ValueError("Only id and destination layer are allowed; text cannot be edited")
        key, layer = assignment["id"], assignment["layer"]
        if not isinstance(key, str) or key not in temporary or key in plan or layer not in LAYERS:
            raise ValueError("Unknown/duplicate record or invalid layer")
        if layer == "L3" and temporary[key]["kind"] != "preference":
            raise ValueError("Only explicitly approved preferences can enter L3")
        plan[key] = layer
    if set(plan) != set(temporary):
        raise ValueError("Plan must account for every eligible temporary record")
    return plan
