"""Injected-transport chat bridge scaffold. No server, networking or inference here."""
import copy
import io
import json

from .store import text
from .transcript import visible_transcript


def prepare_request(model, messages, tools=None):
    text(model, 200)
    messages, pending = visible_transcript(messages)
    if pending:
        raise ValueError("Resolve pending tool results before requesting another model turn")
    request = {"model": model, "messages": messages, "stream": True}
    if tools:
        if not isinstance(tools, list) or len(tools) > 32:
            raise ValueError("Expected a bounded selected tool surface")
        names = set()
        for tool in tools:
            if not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool["type"] != "function":
                raise ValueError("Expected wrapped function schemas")
            function = tool["function"]
            if not isinstance(function, dict) or not {"name", "parameters"} <= set(function):
                raise ValueError("Missing function schema")
            name = text(function["name"], 200)
            if name in names or not isinstance(function["parameters"], dict):
                raise ValueError("Duplicate tool or invalid parameters")
            names.add(name)
        if len(json.dumps(tools).encode("utf-8")) > 64000:
            raise ValueError("Selected schema surface exceeds bridge budget")
        request["tools"] = copy.deepcopy(tools)
    return request


class StreamCollector:
    """One choice only; tool fragments are assembled but NEVER dispatched."""
    def __init__(self):
        self.content = ""
        self.calls = {}
        self.finish = None

    def feed(self, chunk):
        if not isinstance(chunk, dict) or chunk.get("error"):
            raise ValueError("Backend returned an invalid/error response")
        choices = chunk.get("choices")
        if choices == [] and "usage" in chunk:
            return
        if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                or type(choices[0].get("index")) is not int or choices[0]["index"] != 0):
            raise ValueError("Expected one streaming choice")
        choice = choices[0]
        if self.finish is not None:
            raise ValueError("Backend sent data after finish")
        delta = choice.get("delta")
        if not isinstance(delta, dict) or set(delta) - {"role", "content", "tool_calls"}:
            raise ValueError("Unsupported streaming delta")
        if delta.get("role", "assistant") != "assistant":
            raise ValueError("Invalid stream role")
        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise ValueError("Expected text stream")
            self.content += content
        fragments = delta.get("tool_calls") or []
        if not isinstance(fragments, list) or len(fragments) > 32:
            raise ValueError("Expected bounded tool fragments")
        for fragment in fragments:
            self._tool(fragment)
        if len(self.content) + sum(len(c["function"]["arguments"]) for c in self.calls.values()) > 64000:
            raise ValueError("Backend response exceeds bridge budget")
        finish = choice.get("finish_reason")
        if finish is not None:
            if finish not in {"stop", "tool_calls", "length"}:
                raise ValueError("Unsupported finish reason")
            self.finish = finish

    def _tool(self, fragment):
        if not isinstance(fragment, dict) or set(fragment) - {"index", "id", "type", "function"}:
            raise ValueError("Invalid tool fragment")
        index = fragment.get("index")
        if type(index) is not int or not 0 <= index < 32:
            raise ValueError("Invalid tool index")
        call = self.calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        if fragment.get("type", "function") != "function":
            raise ValueError("Unsupported tool type")
        if "id" in fragment:
            if call["id"] or not isinstance(fragment["id"], str):
                raise ValueError("Duplicate/invalid tool ID fragment")
            call["id"] = text(fragment["id"], 200)
        function = fragment.get("function", {})
        if not isinstance(function, dict) or set(function) - {"name", "arguments"}:
            raise ValueError("Invalid streamed function")
        for key, value in function.items():
            if not isinstance(value, str):
                raise ValueError("Function fragments must be text")
            call["function"][key] += value
        if len(call["function"]["name"]) > 200 or len(call["function"]["arguments"]) > 64000:
            raise ValueError("Function fragments exceed bridge budget")

    def result(self, allowed_tools=()):
        if self.finish is None:
            raise ValueError("Interrupted stream has no successful completion")
        if self.finish == "length":
            raise ValueError("Truncated response cannot dispatch tools or claim completion")
        if bool(self.calls) != (self.finish == "tool_calls"):
            raise ValueError("Finish reason does not match tool envelopes")
        calls = [self.calls[key] for key in sorted(self.calls)]
        if sorted(self.calls) != list(range(len(calls))):
            raise ValueError("Missing tool fragment index")
        ids = set()
        from local_extensions.e4b_http import read_json
        for call in calls:
            key, function = text(call["id"], 200), call["function"]
            if key in ids or function["name"] not in allowed_tools:
                raise ValueError("Duplicate ID or tool not in selected schema surface")
            ids.add(key)
            raw = function["arguments"].encode("utf-8")
            if not isinstance(read_json(io.BytesIO(raw), len(raw)), dict):
                raise ValueError("Tool arguments must be a JSON object")
        message = {"role": "assistant", "content": self.content or None}
        if calls:
            message["tool_calls"] = copy.deepcopy(calls)
        return {"message": message, "finish_reason": self.finish, "tools_dispatched": False}


def scripted_completion(request, transport, cancelled=lambda: False):
    """Transport owns real I/O, timeouts and hard cancellation; this checks between chunks."""
    if cancelled():
        raise InterruptedError("Request cancelled")
    collector = StreamCollector()
    for chunk in transport(copy.deepcopy(request)):
        if cancelled():
            raise InterruptedError("Stream cancelled; no tools dispatched")
        collector.feed(chunk)
    if cancelled():
        raise InterruptedError("Completion cancelled")
    names = [tool["function"]["name"] for tool in request.get("tools", [])]
    return collector.result(names)
