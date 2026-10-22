"""Conservative visible-text transcript contract. No model or tokenizer required."""
import copy
import json

from .step_policy import reject_sensitive_text
from .step_recovery import digest
from .store import text

MAX_MESSAGES = 200
MAX_TRANSCRIPT_BYTES = 128000


def visible_transcript(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError("Expected 1–200 visible messages; no history was truncated")
    result, pending, seen = [], set(), set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Expected message objects")
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError("Unsupported message role")
        allowed = {"role", "content", "name"}
        allowed |= {"tool_calls"} if role == "assistant" else set()
        allowed |= {"tool_call_id"} if role == "tool" else set()
        if set(message) - allowed:
            raise ValueError("Unsupported message fields; hidden reasoning/media cannot be archived")
        if pending and role != "tool":
            raise ValueError("Tool-call/result sequence is incomplete")
        content = message.get("content")
        if content is not None:
            text(content, 16000 if role == "assistant" else 4000)
        elif role != "assistant" or not message.get("tool_calls"):
            raise ValueError("Only text messages and assistant tool envelopes are supported")
        if "name" in message:
            text(message["name"], 200)
        if role == "assistant" and "tool_calls" in message:
            _calls(message["tool_calls"], pending, seen)
        if role == "tool":
            key = message.get("tool_call_id")
            if not isinstance(key, str) or key not in pending:
                raise ValueError("Orphan or duplicate tool result")
            pending.remove(key)
        result.append(copy.deepcopy(message))
    reject_sensitive_text(result)
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("Visible checkpoint exceeds budget; original context must be retained")
    return result, sorted(pending)


def _calls(calls, pending, seen):
    if not isinstance(calls, list) or not 1 <= len(calls) <= 32:
        raise ValueError("Expected bounded tool calls")
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"} or call["type"] != "function":
            raise ValueError("Expected function-call envelope")
        key, function = text(call["id"], 200), call["function"]
        if key != call["id"] or key in seen:
            raise ValueError("Duplicate/noncanonical tool-call ID")
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise ValueError("Expected function name and serialized arguments")
        text(function["name"], 200)
        text(function["arguments"], 16000)
        pending.add(key)
        seen.add(key)


def source_events(messages):
    """Each complete user clause stays intact; tool output never defines a requirement."""
    events = []
    for index, message in enumerate(messages):
        if message["role"] not in {"user", "tool"}:
            continue
        content = message["content"]
        outcome = "unknown"
        if message["role"] == "tool":
            try:
                value = json.loads(content)
            except ValueError:
                value = None
            if isinstance(value, dict) and type(value.get("success")) is bool:
                outcome = "reported_success" if value["success"] else "reported_failure"
        events.append({"id": f"event-{index}-{digest(message)[:20]}", "role": message["role"],
                       "content": content, "tool_call_id": message.get("tool_call_id", ""),
                       "outcome": outcome})
    return events
