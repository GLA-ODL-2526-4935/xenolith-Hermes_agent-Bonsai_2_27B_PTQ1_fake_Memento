"""Single-owner, text-only memory. The controller API is not a model tool."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from .maintenance import MaintenanceMixin
from .step_recovery import StepRecoveryMixin

LAYERS = ("L0", "L1", "L2", "L3")
KINDS = ("success", "fact", "decision", "preference", "critical_failure")
MAX_BYTES = 32 * 1024 * 1024


def text(value, maximum=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("Expected nonempty bounded text")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise ValueError("Control/binary characters are forbidden")
    if "data:" in value.lower() or "base64," in value.lower():
        raise ValueError("Inline media is forbidden")
    return value.strip()


@contextmanager
def file_lock(path):
    """OS releases the lock on crash; the stable lock file is never replaced."""
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class Store(MaintenanceMixin, StepRecoveryMixin):
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "memory.json"

    def _read(self):
        if not self.path.exists():
            return {"version": 1, "revision": 0, "records": [], "stages": {},
                    "active": {}, "last_activity": 0, "tasks": {}, "requests": {}}
        if self.path.stat().st_size > MAX_BYTES:
            raise ValueError("Memory store exceeds size limit; owner maintenance required")
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("version") != 1:
            raise ValueError("Unsupported memory format")
        state.setdefault("tasks", {})
        state.setdefault("requests", {})
        for record in state["records"]:
            validate_record(record)
        return state

    def _write(self, state):
        state["revision"] += 1
        payload = json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)
        if len(payload.encode("utf-8")) > MAX_BYTES:
            raise ValueError("Memory full; owner maintenance required")
        fd, name = tempfile.mkstemp(dir=self.root, prefix=".memory-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
            if os.name != "nt":
                directory_fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @contextmanager
    def transaction(self):
        with file_lock(self.root / "memory.lock"):
            state = self._read()
            before = copy.deepcopy(state)
            yield state
            if state != before:
                self._write(state)

    def snapshot(self):
        with self.transaction() as state:
            return copy.deepcopy(state)

    def activity(self, session, active, now=None):
        now = time.time() if now is None else now
        with self.transaction() as state:
            if active:
                state["active"][session] = now
            else:
                state["active"].pop(session, None)
            state["last_activity"] = now

    def open_checkpoint(self, session, evidence, *, now=None, ttl=300, budget=8):
        """OWNER ONLY: admit verified evidence, never trust a model's verified=true."""
        text(session, 200)
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
            raise ValueError("Provide 1–8 owner-verified evidence items")
        if not 1 <= ttl <= 900 or not 1 <= budget <= 20:
            raise ValueError("Checkpoint TTL/budget out of range")
        approved = {}
        for item in evidence:
            if set(item) != {"id", "kind", "text"} or item["kind"] not in KINDS:
                raise ValueError("Evidence requires id, approved kind and text")
            key = text(item["id"], 100)
            if key in approved:
                raise ValueError("Duplicate evidence ID")
            approved[key] = {"kind": item["kind"], "text": text(item["text"], 1200)}
        now = time.time() if now is None else now
        with self.transaction() as state:
            state["stages"][session] = {"expires": now + ttl, "remaining": budget,
                                         "evidence": approved}
            state["last_activity"] = now

    def checkpoint(self, session, now=None):
        now = time.time() if now is None else now
        stage = self.snapshot()["stages"].get(session)
        return stage if stage and stage["expires"] > now and stage["remaining"] > 0 else None

    def append(self, session, *, layer, kind, label, content, evidence_ids, now=None):
        now = time.time() if now is None else now
        candidate = {"layer": layer, "kind": kind, "label": text(label, 120),
                     "content": text(content), "bucket": "temporary"}
        validate_record(candidate)
        if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 5:
            raise ValueError("Provide 1–5 approved evidence IDs")
        if any(not isinstance(key, str) for key in evidence_ids):
            raise ValueError("Invalid evidence ID")
        with self.transaction() as state:
            stage = state["stages"].get(session)
            if not stage or stage["expires"] <= now or stage["remaining"] <= 0:
                raise PermissionError("No open controller-authorized checkpoint")
            evidence = stage["evidence"]
            if any(key not in evidence or evidence[key]["kind"] != kind for key in evidence_ids):
                raise PermissionError("Evidence absent or not approved for this record kind")
            sources = [{"id": key, **evidence[key]} for key in dict.fromkeys(evidence_ids)]
            candidate.update(id=uuid.uuid4().hex, created=now, session=session, evidence=sources)
            duplicate = next((r for r in state["records"] if fingerprint(r) == fingerprint(candidate)), None)
            if duplicate:
                return duplicate["id"]
            state["records"].append(candidate)
            stage["remaining"] -= 1
            state["last_activity"] = now
            return candidate["id"]

    def consolidate(self, *, idle_seconds=10800, now=None):
        """Exact promotion, no LLM rewriting. One atomic snapshot includes clearing temp."""
        if idle_seconds < 60:
            raise ValueError("Idle interval must be at least 60 seconds")
        now = time.time() if now is None else now
        with self.transaction() as state:
            if state["active"] or now - state["last_activity"] < idle_seconds:
                return 0
            count = 0
            for record in state["records"]:
                if record["bucket"] == "temporary":
                    record["bucket"] = "persistent"
                    record["consolidated"] = now
                    count += 1
            state["stages"] = {s: g for s, g in state["stages"].items() if g["expires"] > now}
            return count

    def owner_edit(self, record_id, *, content=None, delete=False):
        with self.transaction() as state:
            record = next((r for r in state["records"] if r["id"] == record_id), None)
            if record is None:
                raise ValueError("Unknown record")
            if delete:
                state["records"].remove(record)
            else:
                record["content"] = text(content)
                record["owner_edited"] = time.time()


def validate_record(record):
    if record["layer"] not in LAYERS or record["kind"] not in KINDS:
        raise ValueError("Unsupported layer or kind (routine failures are not admitted)")
    if record["bucket"] not in ("temporary", "persistent"):
        raise ValueError("Invalid bucket")
    if record["layer"] == "L3" and record["kind"] != "preference":
        raise ValueError("L3 requires an explicitly confirmed preference")
    text(record["content"])
    text(record["label"], 120)


def fingerprint(record):
    value = [record[key] for key in ("layer", "kind", "label", "content")]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()
