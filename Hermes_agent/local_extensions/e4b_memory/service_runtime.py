"""Service-side surfaces, checkpoint gates and lifecycle. No owner operation on wire."""
import json
import re

from .service_auth import Credentials
from .sorting import request_sort
from .store import Store, text
from .surface import Surface


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("Expected bounded session/instance identifier")
    return value


class Runtime:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        if not (self.root / "credentials.sqlite3").is_file():
            raise PermissionError("Initialize the service explicitly first")
        self.credentials = Credentials(self.root)
        self.store = Store(self.root / "memory")
        self.surfaces = {}
        self.pending = {}

    def dispatch(self, token, body):
        client, role = self.credentials.resolve(token)
        if not isinstance(body, dict) or set(body) != {"session", "instance", "action", "arguments"}:
            raise ValueError("Invalid memory request")
        session = client + ":" + identifier(body["session"])
        instance = identifier(body["instance"])
        key = session + ":" + instance
        if key not in self.surfaces and role == "controller":
            if len(self.surfaces) >= 128:
                raise PermissionError("Session capacity reached; owner restart/maintenance required")
            self.surfaces[key] = Surface(self.store, session)
        if key not in self.surfaces:
            raise PermissionError("Controller must establish this session first")
        action, values = body["action"], body["arguments"]
        if not isinstance(values, dict):
            raise ValueError("Expected arguments object")
        if role == "model":
            if action != "memory":
                raise PermissionError("Model credential permits only the memory bridge")
            return self.surfaces[key].dispatch(values)
        handler = CONTROLLER.get(action)
        if handler is None:
            raise PermissionError("Controller cannot approve, edit, delete or read the store")
        return handler(self, key, session, values)

    def begin(self, key, session, values):
        if set(values) != {"message"}:
            raise ValueError("Expected only message")
        text(values["message"], 1200)
        if key not in self.pending and key in self.store.snapshot()["active"]:
            raise PermissionError("Abandoned activity marker; owner recovery is required")
        self.pending[key] = self.pending.get(key, 0) + 1
        self.store.activity(key, True)
        self.store.task_start(session, values["message"])
        self.surfaces[key].reset_budget()
        return {"started": True}

    def capture(self, key, session, values, *, compress=False):
        if set(values) != {"candidates", "error"}:
            raise ValueError("Expected bounded tool assertions only")
        candidates = values["candidates"]
        if not isinstance(candidates, list) or len(candidates) > 8:
            raise ValueError("Too many review candidates")
        messages = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {"kind", "text"}:
                raise ValueError("Invalid candidate")
            kind = candidate["kind"]
            if kind not in {"success", "critical_failure"}:
                raise ValueError("Unsupported evidence kind")
            result = {"success": kind == "success", "critical_failure": kind == "critical_failure",
                      "assertion": text(candidate["text"], 1200)}
            messages.append({"role": "tool", "content": json.dumps(result)})
        error = values["error"]
        if error:
            messages.append({"role": "tool", "content": json.dumps({"success": False, "error": text(error, 800)})})
        elif not isinstance(error, str):
            raise ValueError("Expected error text")
        self.store.capture(session, messages, reason="context_pressure" if compress else None)
        if compress:
            self.surfaces[key] = Surface(self.store, session)
        return {"review_candidates_saved": True, "verified": False}

    def finish(self, key, session, values):
        self.capture(key, session, values)
        self.store.finish_task_turn(session)
        self.pending[key] = max(0, self.pending.get(key, 0) - 1)
        self.store.activity(key, self.pending[key] > 0)
        return {"finished": True}

    def close(self, key, session, values):
        if values:
            raise ValueError("Close has no arguments")
        self.pending.pop(key, None)
        self.store.activity(key, False)
        self.surfaces.pop(key, None)
        return {"closed": True}

    def save_step(self, key, session, values):
        if self.pending.get(key, 0) < 1:
            raise PermissionError("Controller must begin this turn before saving a recovery step")
        return self.store.save_step(session, values)

    def shadow(self, key, session, values):
        if self.pending.get(key, 0) < 1:
            raise PermissionError("Begin a turn before shadow capture")
        if set(values) != {"messages"}:
            raise ValueError("Expected only visible messages")
        from .checkpoint import try_shadow
        return try_shadow(self.store, session, values["messages"])

    def maintain(self):
        settings = json.loads((self.root / "settings.json").read_text(encoding="utf-8"))
        self.store.prepare_idle_checkpoints()
        if settings["semantic_sort"]:
            snapshot = self.store.maintenance_snapshot()
            if snapshot:
                plan = request_sort(snapshot, settings["base_url"], settings["model"])
                self.store.apply_sort_plan(snapshot["revision"], plan)
        else:
            self.store.consolidate()
        return {"expired": self.store.expire(l0_days=settings["l0_days"], recovery_days=settings["recovery_days"])}


CONTROLLER = {"begin": Runtime.begin, "finish": Runtime.finish, "capture": Runtime.capture,
              "shadow": Runtime.shadow,
              "save_step": Runtime.save_step,
              "compress": lambda runtime, key, session, values: runtime.capture(key, session, values, compress=True),
              "close": Runtime.close}


def create(root):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    Credentials(root).initialize()
    settings = {"semantic_sort": False, "l0_days": 30, "recovery_days": 7, "base_url": "", "model": ""}
    (root / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return Runtime(root)
