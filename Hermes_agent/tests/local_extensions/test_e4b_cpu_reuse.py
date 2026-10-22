"""Opt-in prefix reuse retains exact sources and never replays uncertain POSTs."""
import io
import argparse
import json

import pytest

from local_extensions.e4b_memory.controlled_steps import StepController
from local_extensions.e4b_memory.controlled_runtime import LocalJSONModel, READ_TOOL


def plan():
    return {"short_plan": "Read sources", "steps": [
        {"id": "s1", "goal": "Read first source"}, {"id": "s2", "goal": "Read second source"}]}


def test_cache_layout_preserves_evidence_corrections_and_shared_prefix(tmp_path):
    replies = iter([plan(), {"action": "call", "step_id": "s1", "tool": "read_text", "arguments": {"path": "a"}}])
    c = StepController(tmp_path / "cached", "Exact task <= 10 ms", READ_TOOL,
                       lambda messages: json.dumps(next(replies)),
                       lambda call: {"status": "reported_success", "output": "Exact source <= 3 kg"},
                       reuse_prefix=True)
    c.tick()
    first = c.messages(c.journal.read())
    c.tick()
    second = c.messages(c.journal.read())
    stable = second[1]["content"].split('"action_history":', 1)[0]
    assert first[1]["content"].startswith(stable)
    assert [m["role"] for m in second] == ["system", "user"]
    assert json.loads(second[1]["content"])["action_history"][0]["outcome"]["output"] == "Exact source <= 3 kg"
    c.reuse_prefix = False
    ordinary = json.loads(c.messages(c.journal.read())[1]["content"])
    assert ordinary == json.loads(second[1]["content"])
    c.add_user("Correction: stop after inspecting; do not write.")
    c.reuse_prefix = True
    corrected = json.loads(c.messages(c.journal.read())[1]["content"])
    assert corrected["user_instructions_in_order"] == ["Exact task <= 10 ms", "Correction: stop after inspecting; do not write."]
    assert all("reasoning" not in m for m in second)


class Reply(io.BytesIO):
    status, reason = 200, "OK"


class Connection:
    def __init__(self, *args, **kwargs):
        self.requests, self.closed = [], False

    def request(self, method, path, body, headers):
        self.requests.append(json.loads(body))

    def getresponse(self):
        return Reply(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(plan()), "reasoning_content": "MUST NEVER BE CARRIED"}}]}).encode())

    def close(self):
        self.closed = True


def test_local_transport_reuses_connection_but_not_reasoning(tmp_path, monkeypatch):
    connection = Connection()
    created = []
    def factory(*args, **kwargs):
        created.append(args)
        return connection
    monkeypatch.setattr("http.client.HTTPConnection", factory)
    model = LocalJSONModel("http://127.0.0.1:18780", "local", reuse_prefix=True, keep_alive=True)
    c = StepController(tmp_path / "run", "Read source", READ_TOOL, model, lambda call: None, reuse_prefix=True)
    messages = c.messages(c.journal.read())
    assert json.loads(model(messages)) == plan()
    assert json.loads(model(messages)) == plan()
    assert len(created) == 1 and len(connection.requests) == 2
    for payload in connection.requests:
        assert payload["cache_prompt"] is True and payload["id_slot"] == 0
        assert "MUST NEVER BE CARRIED" not in json.dumps(payload)
    model.close()
    assert connection.closed


def test_disconnected_post_is_not_replayed_or_dispatched(tmp_path, monkeypatch):
    connection = Connection()
    def disconnected(*args):
        connection.requests.append("uncertain POST")
        raise ConnectionResetError("Disconnected after sending")
    connection.request = disconnected
    monkeypatch.setattr("http.client.HTTPConnection", lambda *args, **kwargs: connection)
    model = LocalJSONModel("http://127.0.0.1:18780", "local", keep_alive=True)
    actions = []
    c = StepController(tmp_path / "run", "Read source", READ_TOOL, model, actions.append)
    with pytest.raises(ConnectionResetError):
        c.run()
    assert connection.requests == ["uncertain POST"] and connection.closed
    assert actions == [] and c.journal.read()["phase"] == "stopped_error"


def test_redirect_response_is_rejected_without_replay(tmp_path, monkeypatch):
    connection = Connection()
    def redirect():
        response = Reply(b"")
        response.status, response.reason = 302, "Moved"
        return response
    connection.getresponse = redirect
    monkeypatch.setattr("http.client.HTTPConnection", lambda *args, **kwargs: connection)
    model = LocalJSONModel("http://127.0.0.1:18780", "local", keep_alive=True)
    c = StepController(tmp_path / "run", "Read source", READ_TOOL, model, lambda call: None)
    with pytest.raises(ValueError, match="302"):
        model(c.messages(c.journal.read()))
    assert len(connection.requests) == 1 and connection.closed


def test_owner_cli_options_reach_real_controller_and_reader(tmp_path, monkeypatch):
    from local_extensions.e4b_memory.cli import register_cli
    from local_extensions.e4b_memory.controlled_runtime import run_controlled
    task = tmp_path / "task.txt"
    task.write_text("Read evidence.txt and report its exact value", encoding="utf-8")
    (tmp_path / "evidence.txt").write_text("Exact value: 17 kg", encoding="utf-8")
    parser = argparse.ArgumentParser()
    register_cli(parser)
    args = parser.parse_args(["controlled-run", str(task), "--session", "isolated", "--workspace", str(tmp_path),
                             "--base-url", "http://127.0.0.1:18780", "--model", "local",
                             "--reuse-prefix", "--keep-alive"])
    initial = plan()
    initial["steps"] = initial["steps"][:1]
    replies = iter([initial, {"action": "call", "step_id": "s1", "tool": "read_text", "arguments": {"path": "evidence.txt"}},
                    {"action": "finish", "answer": "Reported value: 17 kg", "unfinished": []}])
    connection = Connection()
    def response():
        return Reply(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(next(replies)), "reasoning_content": "MUST NEVER BE CARRIED"}}]}).encode())
    connection.getresponse = response
    monkeypatch.setattr("http.client.HTTPConnection", lambda *args, **kwargs: connection)
    state = run_controlled(args, tmp_path / "owner-home")
    assert state["phase"] == "completed" and state["unfinished"] == []
    assert state["results"]["call-1"]["output"] == "Exact value: 17 kg"
    assert len(connection.requests) == 3 and connection.closed
    assert all(p["cache_prompt"] and p["id_slot"] == 0 for p in connection.requests)
    assert "MUST NEVER BE CARRIED" not in json.dumps(connection.requests)
    assert (tmp_path / "owner-home/controlled_runs/isolated/working-context.json").exists()
