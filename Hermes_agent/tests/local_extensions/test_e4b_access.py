"""CPU-only broker contracts: never start a browser, Hermes agent or subprocess."""
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from local_extensions.e4b_access.broker import Broker
from local_extensions.e4b_access.client import Client, NoRedirect
from local_extensions.e4b_access.policy import public_https_url, validate_proposal
from local_extensions.e4b_access.service import dispatch, handler_for
from local_extensions.e4b_access.surface import dispatch as bridge


@pytest.fixture
def inbox(tmp_path):
    return Broker.create(tmp_path / "broker", clock=lambda: 100)


def proposal(**updates):
    return {"kind": "text_export", "label": "Report", "text": "Plain report", **updates}


def test_export_needs_owner_and_is_immutable(inbox):
    broker, token = inbox
    result = dispatch(broker, token, {"action": "submit", "proposal": proposal()})
    assert list((broker.root / "exports").iterdir()) == []
    with pytest.raises(PermissionError):
        dispatch(broker, token, {"action": "approve", "id": result["id"], "digest": result["digest"]})
    approved = broker.decide(result["id"], result["digest"], approve=True)
    assert Path(approved["path"]).read_text() == "Plain report"
    assert Path(approved["path"]).name == result["id"] + ".txt"
    with pytest.raises(PermissionError):
        broker.decide(result["id"], result["digest"], approve=True)


def test_no_model_paths_or_capability_escalation():
    for payload in (proposal(path="../../memory.json"), proposal(filename="run.py"),
                    {"kind": "shell", "command": "anything"}, proposal(kind="delete")):
        with pytest.raises((ValueError, PermissionError)):
            validate_proposal(payload)


def test_label_is_not_a_file_path(inbox):
    broker, _token = inbox
    result = broker.submit(proposal(label="../../config.yaml"))
    approved = broker.decide(result["id"], result["digest"], approve=True)
    assert Path(approved["path"]).parent == broker.root / "exports"
    assert Path(approved["path"]).suffix == ".txt"


def test_approval_binds_content_and_expiry(inbox):
    broker, _token = inbox
    result = broker.submit(proposal())
    with pytest.raises(PermissionError):
        broker.decide(result["id"], "different", approve=True)
    broker.clock = lambda: result["expires"]
    assert broker.status(result["id"])["status"] == "expired"
    assert broker.list_pending() == []
    with pytest.raises(PermissionError):
        broker.decide(result["id"], result["digest"], approve=True)


def test_rejection_does_not_write(inbox):
    broker, _token = inbox
    result = broker.submit(proposal())
    assert broker.decide(result["id"], result["digest"], approve=False)["status"] == "rejected"
    assert list((broker.root / "exports").iterdir()) == []


def test_existing_export_is_never_overwritten(inbox):
    broker, _token = inbox
    result = broker.submit(proposal())
    target = broker.root / "exports" / (result["id"] + ".txt")
    target.write_text("Owner content", encoding="utf-8")
    with pytest.raises(PermissionError):
        broker.decide(result["id"], result["digest"], approve=True)
    assert target.read_text() == "Owner content"
    assert broker.status(result["id"])["status"] == "pending"


def test_restart_and_write_before_commit_recovery(inbox):
    broker, _token = inbox
    result = broker.submit(proposal())
    target = broker.root / "exports" / (result["id"] + ".txt")
    target.write_text("Plain report", encoding="utf-8")
    recovered = Broker(broker.root, clock=lambda: 101)
    assert recovered.decide(result["id"], result["digest"], approve=True)["status"] == "exported"


def test_browser_approval_is_manual_handoff_only(inbox):
    broker, _token = inbox
    for kind in ("open_page", "download", "upload", "submit_form", "send_message", "purchase", "account_change"):
        result = broker.submit({"kind": kind, "url": "https://example.com/file", "details": "Exact owner-reviewed action"})
        approved = broker.decide(result["id"], result["digest"], approve=True)
        assert approved["status"] == "approved_manual"
        assert approved["url"] == "https://example.com/file"
    assert list((broker.root / "exports").iterdir()) == []


def test_private_or_ambiguous_urls_refused():
    for url in ("file:///secrets", "javascript:alert(1)", "http://example.com", "https://localhost",
                "https://127.0.0.1", "https://[::1]", "https://192.168.1.1", "https://169.254.169.254",
                "https://name.local", "https://user:pass@example.com", "https://example.com:8443",
                "https://2130706433", "https://127.1", "https://example.com\\@localhost", "https://example.com/\n"):
        with pytest.raises((ValueError, PermissionError)):
            public_https_url(url)
    assert public_https_url("https://EXAMPLE.COM") == "https://example.com/"


def test_credentials_and_proposal_only_transport(inbox):
    broker, token = inbox
    for credential in (None, "", "x" * 43):
        with pytest.raises(PermissionError):
            dispatch(broker, credential, {"action": "submit", "proposal": proposal()})
    result = dispatch(broker, token, {"action": "submit", "proposal": proposal()})
    assert dispatch(broker, token, {"action": "status", "id": result["id"]})["status"] == "pending"
    for action in ("list", "inspect", "delete", "approve", "execute", "read_file"):
        with pytest.raises(PermissionError):
            dispatch(broker, token, {"action": action})


def test_capacity_bounds_and_id_validation(inbox):
    broker, _token = inbox
    for content in ("", "x" * 65537, "\x00", "é" * 32769):
        with pytest.raises(ValueError):
            broker.submit(proposal(text=content))
    for key in ("../../file", "' OR 1=1", "a" * 32):
        with pytest.raises(ValueError):
            broker.status(key)
    with broker.transaction() as db:
        db.executemany("INSERT INTO requests VALUES (?, '{}', 'hash', 0, 1, 'expired')",
                       [(str(n),) for n in range(1000)])
    with pytest.raises(PermissionError):
        broker.submit(proposal())


def test_initialization_refuses_existing_directory(inbox):
    broker, _token = inbox
    with pytest.raises(FileExistsError):
        Broker.create(broker.root)


def test_progressive_discovery_has_no_owner_operations(inbox):
    broker, token = inbox
    client = SimpleNamespace(call=lambda body: dispatch(broker, token, body))
    factory = lambda: client
    assert bridge({"action": "sections"}, factory)["sections"] == ["browser", "files", "requests"]
    catalog = bridge({"action": "catalog", "section": "files"}, factory)
    assert catalog["operations"][0]["name"] == "text_export"
    assert "parameters" not in catalog["operations"][0]
    description = bridge({"action": "describe", "section": "files", "name": "text_export"}, factory)
    assert set(description["parameters"]["properties"]) == {"label", "text"}
    result = bridge({"action": "call", "section": "files", "name": "text_export",
                     "arguments": {"label": "Test", "text": "test"}}, factory)
    assert result["status"] == "pending"
    with pytest.raises(PermissionError):
        bridge({"action": "describe", "section": "requests", "name": "approve"}, factory)
    with pytest.raises(ValueError):
        bridge({"action": "call", "section": "files", "name": "text_export",
                "arguments": {"kind": "download", "label": "test", "text": "test"}}, factory)


def test_client_restricts_routes_redirects_and_proxy_use(inbox):
    _broker, token = inbox
    for endpoint in ("https://example.com", "http://localhost:8080/v1/requests", "http://127.0.0.1:8080/approve"):
        with pytest.raises(ValueError):
            Client(endpoint, token)
    with pytest.raises(PermissionError):
        NoRedirect().redirect_request(None, None, 302, "", {}, "http://example.com")
    response = io.BytesIO(b'{"success":true,"status":"pending"}')
    with patch("local_extensions.e4b_access.client.build_opener") as opener:
        opener.return_value.open.return_value = response
        assert Client("http://127.0.0.1:8080/v1/requests", token).call({"action": "status", "id": "a"})["status"] == "pending"
        assert opener.call_args.args[0].proxies == {}
        request = opener.return_value.open.call_args.args[0]
        assert request.get_header("Authorization") == "Bearer " + token


class FakeSocket:
    def __init__(self, data):
        self.reader = io.BytesIO(data)
        self.output = bytearray()

    def settimeout(self, _timeout):
        return

    def makefile(self, *_args):
        return self.reader

    def sendall(self, data):
        self.output.extend(data)


def http_call(broker, token, body, extra="", path="/v1/requests"):
    payload = json.dumps(body).encode()
    headers = (f"POST {path} HTTP/1.0\r\nHost: 127.0.0.1:8080\r\nContent-Type: application/json\r\n"
               f"Authorization: Bearer {token}\r\nContent-Length: {len(payload)}\r\n{extra}\r\n")
    socket = FakeSocket(headers.encode() + payload)
    handler_for(broker)(socket, ("127.0.0.1", 1), SimpleNamespace(server_port=8080))
    return bytes(socket.output)


def test_real_http_handler_without_network_or_browser(inbox):
    broker, token = inbox
    result = http_call(broker, token, {"action": "submit", "proposal": proposal()})
    assert b"200 OK" in result
    assert json.loads(result.split(b"\r\n\r\n")[1])["status"] == "pending"
    for extra in ("Origin: https://evil.example\r\n", "Transfer-Encoding: chunked\r\n", "Content-Length: 1\r\n"):
        assert b"400 Bad Request" in http_call(broker, token, {}, extra=extra)
    assert b"400 Bad Request" in http_call(broker, token, {}, path="/approve")
    assert b"400 Bad Request" in http_call(broker, "wrong", {})


def test_real_hermes_plugin_discovery():
    from hermes_constants import get_hermes_home
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry
    home = get_hermes_home()
    source = Path(__file__).resolve().parents[2] / "local_extensions" / "e4b_access"
    shutil.copytree(source, home / "plugins" / "e4b_access")
    config = {"plugins": {"enabled": ["e4b_access"], "entries": {"e4b_access": {"settings": {"enabled": False}}}}}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    manager = PluginManager()
    manager.discover_and_load()
    plugin = manager._plugins["e4b_access"]
    assert plugin.enabled, plugin.error
    tool = registry.get_entry("e4b_access", scope=manager.scope_key)
    assert tool is not None
    assert json.loads(tool.handler({"action": "sections"}))["success"] is False


def test_owner_cli_approval_end_to_end(tmp_path, capsys):
    from local_extensions.e4b_access.__main__ import main
    root = str(tmp_path / "owner-inbox")
    main([root, "init"])
    token = json.loads(capsys.readouterr().out)["client_token"]
    broker = Broker(root)
    result = dispatch(broker, token, {"action": "submit", "proposal": proposal()})
    main([root, "inspect", result["id"]])
    reviewed = json.loads(capsys.readouterr().out)
    assert json.loads(reviewed["payload"])["text"] == "Plain report"
    main([root, "approve", result["id"], reviewed["digest"]])
    approved = json.loads(capsys.readouterr().out)
    assert approved["status"] == "exported"
    assert Path(approved["path"]).read_text() == "Plain report"
