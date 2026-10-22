"""Service protocol tests, in-memory transports only; no live agent/model/server."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from local_extensions.e4b_memory.remote import Client, NoRedirect, RemoteMemoryProvider, capture_payload
from local_extensions.e4b_memory.service import main, settings
from local_extensions.e4b_memory.service_http import handler_for
from local_extensions.e4b_memory.service_runtime import Runtime, create


@pytest.fixture
def service(tmp_path):
    runtime = create(tmp_path / "service")
    return runtime, runtime.credentials.issue()


def call(runtime, tokens, action, arguments=None, role="controller", session="session", instance="instance"):
    return runtime.dispatch(tokens[role], {"session": session, "instance": instance,
                                          "action": action, "arguments": arguments or {}})


def begin(runtime, tokens):
    return call(runtime, tokens, "begin", {"message": "Task"})


def memory(runtime, tokens, arguments):
    return call(runtime, tokens, "memory", arguments, role="model")


def describe(runtime, tokens, section, name):
    memory(runtime, tokens, {"action": "catalog", "section": section})
    return memory(runtime, tokens, {"action": "describe", "section": section, "name": name})


def test_authentication_roles_and_no_owner_api(service):
    runtime, tokens = service
    begin(runtime, tokens)
    for role in ("model", "controller"):
        for action in ("approve", "edit", "delete", "open_checkpoint", "export", "issue", "revoke", "maintain"):
            with pytest.raises(PermissionError):
                call(runtime, tokens, action, role=role)
    with pytest.raises(PermissionError):
        call(runtime, tokens, "begin", {"message": "forged"}, role="model")
    with pytest.raises(PermissionError):
        call(runtime, tokens, "memory", {"action": "sections"})
    with pytest.raises(PermissionError):
        runtime.credentials.resolve("x" * 43)


def test_model_cannot_create_session_or_return_full_store(service):
    runtime, tokens = service
    with pytest.raises(PermissionError):
        memory(runtime, tokens, {"action": "sections"})
    begin(runtime, tokens)
    assert "temporary.L0" in memory(runtime, tokens, {"action": "sections"})["sections"]
    with pytest.raises(ValueError):
        memory(runtime, tokens, {"action": "export", "section": "persistent.L1"})


def test_append_gate_owned_by_service(service):
    runtime, tokens = service
    begin(runtime, tokens)
    describe(runtime, tokens, "checkpoint", "append")
    args = {"action": "call", "section": "checkpoint", "name": "append", "arguments": {
        "layer": "L1", "kind": "success", "label": "Task", "content": "Verified success", "evidence_ids": ["owner-proof"]}}
    with pytest.raises(PermissionError):
        memory(runtime, tokens, args)
    session = tokens["client_id"] + ":session"
    runtime.store.open_checkpoint(session, [{"id": "owner-proof", "kind": "success", "text": "Owner verified result"}])
    result = memory(runtime, tokens, args)
    assert runtime.store.snapshot()["records"][0]["id"] == result["id"]
    with pytest.raises(ValueError):
        describe(runtime, tokens, "checkpoint", "delete")


def test_review_candidates_are_never_self_approved(service):
    runtime, tokens = service
    begin(runtime, tokens)
    payload = {"candidates": [{"kind": "success", "text": "Tool asserts task succeeded"}], "error": ""}
    call(runtime, tokens, "compress", payload)
    session = tokens["client_id"] + ":session"
    state = runtime.store.snapshot()
    assert state["requests"][session]["approved"] is False
    assert state["records"] == [] and state["stages"] == {}
    with pytest.raises(ValueError):
        call(runtime, tokens, "capture", {"candidates": [{"kind": "success", "text": "claim", "verified": True}], "error": ""})


def test_finish_captures_recovery_and_delayed_sync_counts(service):
    runtime, tokens = service
    begin(runtime, tokens)
    begin(runtime, tokens)
    payload = {"candidates": [], "error": "Tool argument error"}
    call(runtime, tokens, "finish", payload)
    assert runtime.store.snapshot()["active"]
    call(runtime, tokens, "finish", payload)
    state = runtime.store.snapshot()
    assert state["active"] == {}
    assert state["tasks"][tokens["client_id"] + ":session"]["last_error"] == "Tool argument error"
    assert state["records"] == []


def test_revocation_and_crash_fail_closed(service):
    runtime, tokens = service
    begin(runtime, tokens)
    restarted = Runtime(runtime.root)
    with pytest.raises(PermissionError):
        begin(restarted, tokens)
    runtime.credentials.revoke(tokens["client_id"])
    with pytest.raises(PermissionError):
        memory(runtime, tokens, {"action": "sections"})
    assert runtime.store.snapshot()["active"]  # Owner recovery, never silent idle promotion.


def test_controller_cannot_change_settings_or_supplement_binary(service):
    runtime, tokens = service
    begin(runtime, tokens)
    for payload in ({"candidates": [], "error": "", "l0_days": 1},
                    {"candidates": [{"kind": "success", "text": "data:image/png;base64,abc"}], "error": ""}):
        with pytest.raises(ValueError):
            call(runtime, tokens, "capture", payload)


def test_local_capture_does_not_send_reasoning_images_or_own_memory_results():
    messages = [{"role": "assistant", "content": "Hidden reasoning"},
                {"role": "tool", "content": {"image": "pixels"}},
                {"role": "tool", "name": "e4b_memory", "content": '{"success":true}'},
                {"role": "tool", "content": '{"success":true,"value":"verified later"}'}]
    result = capture_payload(messages)
    assert len(result["candidates"]) == 1
    assert "Hidden reasoning" not in json.dumps(result) and "pixels" not in json.dumps(result)


class DirectClient:
    def __init__(self, runtime, token):
        self.runtime, self.token = runtime, token

    def call(self, session, instance, action, arguments):
        return self.runtime.dispatch(self.token, {"session": session, "instance": instance,
                                                  "action": action, "arguments": arguments})


def test_remote_provider_uses_service_without_creating_local_store(service, tmp_path):
    runtime, tokens = service
    clients = (DirectClient(runtime, tokens["model"]), DirectClient(runtime, tokens["controller"]))
    provider = RemoteMemoryProvider()
    home = tmp_path / "agent-home"
    home.mkdir()
    with patch("local_extensions.e4b_memory.remote.clients", return_value=clients):
        provider.initialize("session", hermes_home=home, platform="cli")
    provider.on_turn_start(1, "Task")
    assert json.loads(provider.handle_tool_call("e4b_memory", {"action": "sections"}))["success"]
    provider.on_pre_compress([{"role": "tool", "content": '{"success":true}'}])
    provider.sync_turn("Task", "Not trusted as proof", messages=[])
    provider.on_session_switch("next")
    provider.on_turn_start(2, "Next")
    provider.shutdown()
    assert not (home / "e4b_memory").exists()
    assert runtime.store.snapshot()["active"] == {}


def test_remote_provider_failure_never_falls_back(service, tmp_path):
    provider = RemoteMemoryProvider()
    failing = SimpleNamespace(call=lambda *_args: (_ for _ in ()).throw(OSError("offline")))
    with patch("local_extensions.e4b_memory.remote.clients", return_value=(failing, failing)):
        provider.initialize("session", hermes_home=tmp_path, platform="cli")
    assert json.loads(provider.handle_tool_call("e4b_memory", {"action": "sections"}))["success"] is False
    with pytest.raises(OSError):
        provider.on_turn_start(1, "Task")
    assert not (tmp_path / "e4b_memory").exists()
    with pytest.raises(PermissionError):
        RemoteMemoryProvider().initialize("a", hermes_home=tmp_path, platform="telegram")


def test_remote_transport_bounds_and_redirects(service):
    _runtime, tokens = service
    for url in ("http://localhost:8767/v1/memory", "https://example.com", "http://127.0.0.1:8767/owner"):
        with pytest.raises(ValueError):
            Client(url, tokens["model"])
    with pytest.raises(PermissionError):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")
    with patch("local_extensions.e4b_memory.remote.build_opener") as opener:
        opener.return_value.open.return_value = io.BytesIO(b'{"success":true,"sections":[]}')
        client = Client("http://127.0.0.1:8767/v1/memory", tokens["model"])
        assert client.call("s", "i", "memory", {}) == {"sections": []}
        assert opener.call_args.args[0].proxies == {}


def test_owner_cli_and_service_settings(tmp_path, capsys):
    root = str(tmp_path / "owner-service")
    main([root, "init"])
    capsys.readouterr()
    main([root, "client"])
    tokens = json.loads(capsys.readouterr().out)
    main([root, "owner", "status"])
    assert json.loads(capsys.readouterr().out)["counts"]["temporary.L0"] == 0
    main([root, "revoke", tokens["client_id"]])
    assert json.loads(capsys.readouterr().out)["revoked"] == tokens["client_id"]
    value = {"semantic_sort": False, "l0_days": 40, "recovery_days": 8, "base_url": "", "model": ""}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(value))
    main([root, "configure", str(path)])
    capsys.readouterr()
    assert json.loads((Path(root) / "settings.json").read_text()) == settings(value)
    with pytest.raises(ValueError):
        settings({**value, "semantic_sort": True, "base_url": "https://cloud.example", "model": "wrong"})


def test_server_handler_without_sockets(service):
    from tests.local_extensions.test_e4b_access import FakeSocket
    runtime, tokens = service
    begin(runtime, tokens)
    body = json.dumps({"session": "session", "instance": "instance", "action": "memory",
                       "arguments": {"action": "sections"}}).encode()
    headers = (f"POST /v1/memory HTTP/1.0\r\nHost: 127.0.0.1:8767\r\nContent-Type: application/json\r\n"
               f"Authorization: Bearer {tokens['model']}\r\nContent-Length: {len(body)}\r\n\r\n")
    connection = FakeSocket(headers.encode() + body)
    handler_for(runtime)(connection, ("127.0.0.1", 1), SimpleNamespace(server_port=8767))
    assert b"200 OK" in connection.output
    assert b"temporary.L0" in connection.output


def test_provider_selection_service_and_unknown_backend():
    from local_extensions.e4b_memory import register
    chosen = []
    context = SimpleNamespace(register_memory_provider=chosen.append)
    config = {"memory": {"e4b_memory": {"backend": "service"}}}
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        register(context)
    assert isinstance(chosen[0], RemoteMemoryProvider)
    config["memory"]["e4b_memory"]["backend"] = "typo"
    with patch("hermes_cli.config.load_config_readonly", return_value=config), pytest.raises(ValueError):
        register(context)


def test_restricted_profile_can_select_service_without_changing_tool_permissions(tmp_path):
    from local_extensions.e4b_memory.restricted import create_profile
    profile = tmp_path / "profile"
    create_profile(profile, "http://127.0.0.1:8000/v1", "future-model",
                   memory_service_url="http://127.0.0.1:8767/v1/memory")
    config = json.loads((profile / "config.yaml").read_text())
    assert config["memory"]["e4b_memory"]["backend"] == "service"
    assert config["platform_toolsets"]["cli"] == ["memory"]
    assert "TOKEN" not in json.dumps(config)


def test_old_owner_cli_refuses_to_create_local_store_in_service_mode():
    from hermes_constants import get_hermes_home
    from local_extensions.e4b_memory.cli import run
    config = {"memory": {"e4b_memory": {"backend": "service"}}}
    with patch("hermes_cli.config.load_config_readonly", return_value=config), pytest.raises(PermissionError):
        run(SimpleNamespace(memory_command="status"))
    assert not (get_hermes_home() / "e4b_memory").exists()
