"""Explicit local-only CLI consumer of the controlled-step loop; read-only file sandbox."""
import ipaddress
import http.client
import json
import threading
import time
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from .controlled_state import bounded
from .controlled_steps import StepController

READ_TOOL = {"read_text": {"description": "Read a complete UTF-8 text file under the owner-selected workspace. No commands or writes.",
                           "arguments": {"path": "relative path, required string"}}}


class WorkspaceReader:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Expected an owner-selected directory")

    def __call__(self, call):
        if call["tool"] != "read_text" or set(call["arguments"]) != {"path"}:
            return {"status": "permission_denied", "output": "Only read_text(path) is authorized"}
        relative = bounded(call["arguments"]["path"], 1000)
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            return {"status": "permission_denied", "output": "Path is outside authorized workspace"}
        try:
            if candidate.stat().st_size > 16000:
                return {"status": "reported_failure", "output": "File too large; no truncated evidence returned"}
            output = candidate.read_text(encoding="utf-8")
            bounded(output)
            return {"status": "reported_success", "output": output}
        except (FileNotFoundError, IsADirectoryError, UnicodeError):
            return {"status": "reported_failure", "output": "Requested UTF-8 text file unavailable"}
        except PermissionError:
            return {"status": "permission_denied", "output": "Read denied by filesystem permissions"}


class LocalJSONModel:
    """Strict JSON actions avoid depending on an unvalidated native tool-call template."""
    def __init__(self, base_url, model, max_tokens=1024, timeout=120, *,
                 reuse_prefix=False, keep_alive=False):
        url = urllib.parse.urlsplit(base_url)
        try:
            loopback = ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            loopback = False
        if (url.scheme != "http" or not loopback or url.username or url.password
                or url.query or url.fragment or url.path not in {"", "/", "/v1", "/v1/"}):
            raise ValueError("Use an explicit loopback HTTP inference endpoint; no fallback")
        self.base = base_url.rstrip("/").removesuffix("/v1")
        if type(reuse_prefix) is not bool or type(keep_alive) is not bool:
            raise ValueError("Cache/connection options must be boolean")
        self.reuse_prefix = reuse_prefix
        self.model, self.max_tokens, self.timeout = bounded(model, 1000), max_tokens, timeout
        self.opener = (_KeepAlive(url.hostname, url.port if url.port is not None else 80) if keep_alive else
                       urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()))
        self.usage = []

    def __call__(self, messages):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
                   "repeat_penalty": 1, "seed": 3407, "max_tokens": self.max_tokens,
                   "cache_prompt": self.reuse_prefix, "response_format": {"type": "json_schema", "json_schema": {"name": "controlled_step", "strict": True, "schema": response_schema(messages)}},
                   "chat_template_kwargs": {"enable_thinking": False}}
        if self.reuse_prefix:
            payload["id_slot"] = 0  # Dedicated one-slot server, one foreground controller.
        request = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Model response exceeds budget")
        data = json.loads(raw)
        choices = data["choices"]
        if len(choices) != 1:
            raise ValueError("Expected exactly one model choice")
        self.last_completion = {"finish_reason": choices[0]["finish_reason"],
                                "content": choices[0]["message"].get("content")}
        if choices[0]["finish_reason"] != "stop":
            raise ValueError(f"Incomplete model reply ({choices[0]['finish_reason']}); no action dispatched")
        message = choices[0]["message"]
        if message.get("tool_calls"):
            raise ValueError("Expected controlled JSON protocol, not native tool envelopes")
        self.usage.append(data.get("usage", {}))
        return message["content"]

    def close(self):
        if isinstance(self.opener, _KeepAlive):
            self.opener.close()


class _KeepAlive:
    """One loopback connection; never redirect, proxy or replay an uncertain POST."""
    def __init__(self, host, port):
        self.host, self.port, self.connection = host, port, None

    def open(self, request, timeout):
        if self.connection is None:
            self.connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            path = urllib.parse.urlsplit(request.full_url).path
            self.connection.request("POST", path, request.data, dict(request.header_items()))
            response = self.connection.getresponse()
            if response.status != 200:
                code, reason = response.status, response.reason
                response.close()
                raise ValueError(f"Local inference HTTP {code}: {reason}; no retry")
            return response
        except Exception:
            self.close()
            raise

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Inference redirects are forbidden")


def run_controlled(args, home):
    task = Path(args.task_file).read_text(encoding="utf-8")
    session = bounded(args.session, 80)
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in session):
        raise ValueError("Use an alphanumeric session identifier")
    reuse = getattr(args, "reuse_prefix", False)
    model = LocalJSONModel(args.base_url, args.model, reuse_prefix=reuse,
                          keep_alive=getattr(args, "keep_alive", False))
    controller = StepController(Path(home) / "controlled_runs" / session, task, READ_TOOL,
                                model, WorkspaceReader(args.workspace), reuse_prefix=reuse)
    try:
        state = controller.run()
    finally:
        model.close()
    interval = getattr(args, "idle_handoff_after", None)
    if interval is not None:
        wait_for_handoff(controller.journal, interval)
    return state


def response_schema(messages):
    """Constrain protocol shape, never supply task answers or choose an action."""
    view = json.loads(messages[-1]["content"])
    string = {"type": "string", "minLength": 1}
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    step = obj({"id": string, "goal": string})
    plan = {"short_plan": string, "steps": {"type": "array", "items": step, "minItems": 1, "maxItems": 16}}
    if view["phase"] == "planning":
        return obj(plan)
    choices = [obj({"action": {"const": "finish"}, "answer": string,
                    "unfinished": {"type": "array", "items": string, "maxItems": 32}}),
               obj({"action": {"const": "revise"}, **plan})]
    current = view["current_step"]
    if current:
        choices.append(obj({"action": {"const": "call"}, "step_id": {"const": current["id"]},
                            "tool": {"enum": list(view["allowed_tools"])},
                            "arguments": {"type": "object"}}))
    return {"anyOf": choices}


def wait_for_handoff(journal, idle_seconds, *, clock=time.time, wait=None):
    """Explicit foreground idle timer; caller owns interruption and process lifetime."""
    if type(idle_seconds) is not int or not 0 <= idle_seconds <= 86400:
        raise ValueError("Idle handoff delay must be 0-86400 seconds")
    state = journal.read()
    if state["pending"] or state["phase"] in {"planning", "running", "needs_reconciliation"}:
        return None
    sleeper = wait or threading.Event().wait
    remaining = state["updated"] + idle_seconds - clock()
    while remaining > 0:
        sleeper(min(remaining, 1))
        remaining = journal.read()["updated"] + idle_seconds - clock()
    return journal.idle_handoff(idle_seconds)
