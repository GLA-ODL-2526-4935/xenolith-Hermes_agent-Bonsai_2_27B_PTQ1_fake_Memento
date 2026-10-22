"""Private working-context journal; independent of curated/recovery memory banks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from .step_policy import reject_sensitive_text
from .store import file_lock, text


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def bounded(value, maximum=16000):
    text(value, maximum)
    reject_sensitive_text(value)
    return value


class WorkingJournal:
    """One foreground controller owns a fresh run; restart inspection never replays calls."""
    def __init__(self, root, *, task=None, clock=time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "working-context.json"
        self.clock = clock
        if task is not None:
            with file_lock(self.root / "controller.lock"):
                if self.path.exists():
                    raise FileExistsError("Run already exists; inspect it, never silently replay")
                self.commit({"version": 1, "users": [bounded(task)], "plan": [], "short_plan": "",
                             "cursor": 0, "attempts": [], "results": {}, "pending": None,
                             "phase": "planning", "final": None, "unfinished": [], "revision": 0})
        else:
            self.read()

    def read(self):
        if self.path.stat().st_size > 2_000_000:
            raise ValueError("Working journal exceeds budget")
        wrapper = json.loads(self.path.read_text(encoding="utf-8"))
        state = wrapper["state"]
        digest = hashlib.sha256(encoded(state).encode()).hexdigest()
        if wrapper["digest"] != digest or state["version"] != 1:
            raise ValueError("Working journal integrity failure")
        return copy.deepcopy(state)

    def commit(self, state):
        state = copy.deepcopy(state)
        state["revision"] += 1
        state["updated"] = self.clock()
        reject_sensitive_text(state)
        digest = hashlib.sha256(encoded(state).encode()).hexdigest()
        payload = encoded({"digest": digest, "state": state})
        if len(payload.encode()) > 2_000_000:
            raise ValueError("Working journal full; original state retained")
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".working-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if self.read() != state:
            raise OSError("Working checkpoint readback failed")
        return state

    def idle_handoff(self, idle_seconds=10800):
        """Owner-triggered maintenance; no daemon, promotion or L0-L3 writes."""
        state = self.read()
        if state["pending"] or state["phase"] == "running" or self.clock() - state["updated"] < idle_seconds:
            return None
        result = {"kind": "temporary-controlled-handoff-v1", "users": state["users"],
                  "short_plan": state["short_plan"], "plan": state["plan"],
                  "attempts": state["attempts"], "results": state["results"],
                  "final": state["final"], "unfinished": state["unfinished"],
                  "source_revision": state["revision"], "verified": False,
                  "grants_permission": False}
        # Preserve all outcomes, including unresolved non-permission failures.
        destination = self.root / f"handoff-{state['revision']}.json"
        payload = encoded(result)
        if destination.exists():
            if destination.read_text(encoding="utf-8") != payload:
                raise ValueError("Conflicting handoff")
        else:
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if json.loads(destination.read_text(encoding="utf-8")) != result:
            raise OSError("Handoff readback failed")
        return result
