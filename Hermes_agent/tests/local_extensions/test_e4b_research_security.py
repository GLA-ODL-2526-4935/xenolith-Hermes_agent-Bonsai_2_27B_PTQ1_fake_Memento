"""Offline research restrictions: no model, browser or live sockets."""
import io
import json
import sqlite3

import pytest

from local_extensions.e4b_access.broker import Broker
from local_extensions.e4b_access.policy import WEB_ACTIONS, digest
from local_extensions.e4b_access.surface import dispatch
from local_extensions.e4b_http import DeadlineReader
from local_extensions.e4b_memory.endpoint_policy import EndpointPolicy


def proposal(kind="open_page"):
    if kind == "text_export":
        return {"kind": kind, "label": "test", "text": "test"}
    return {"kind": kind, "url": "https://example.com/", "details": "Research"}


@pytest.mark.parametrize("kind", [*WEB_ACTIONS[1:], "text_export"])
def test_research_broker_denies_nonresearch(tmp_path, kind):
    broker, _ = Broker.create(tmp_path / "broker", scope="research")
    with pytest.raises(PermissionError):
        broker.submit(proposal(kind))
    assert broker.list_pending() == []


def test_research_scope_persists_and_manual_handoff_only(tmp_path):
    broker, token = Broker.create(tmp_path / "broker", scope="research")
    broker = Broker(broker.root)
    broker.authenticate(token)
    request = broker.submit(proposal())
    result = broker.decide(request["id"], request["digest"], approve=True)
    assert result["status"] == "approved_manual"
    assert list((broker.root / "exports").iterdir()) == []


def test_scope_rechecked_on_approval(tmp_path):
    broker, _ = Broker.create(tmp_path / "broker", scope="research")
    request = broker.submit(proposal())
    forbidden = proposal("text_export")
    with sqlite3.connect(broker.database) as db:
        db.execute("UPDATE requests SET payload=?, digest=? WHERE id=?",
                   (json.dumps(forbidden), digest(forbidden), request["id"]))
    with pytest.raises(PermissionError):
        broker.decide(request["id"], digest(forbidden), approve=True)
    assert list((broker.root / "exports").iterdir()) == []


def test_legacy_manual_database(tmp_path):
    broker, _ = Broker.create(tmp_path / "broker")
    with sqlite3.connect(broker.database) as db:
        token_hash = db.execute("SELECT token_hash FROM settings").fetchone()[0]
        db.execute("DROP TABLE settings")
        db.execute("CREATE TABLE settings (token_hash TEXT NOT NULL)")
        db.execute("INSERT INTO settings VALUES (?)", (token_hash,))
    assert Broker(broker.root).scope == "manual"


def test_research_discovery_matches_scope():
    def no_client():
        raise AssertionError("Discovery must not contact a service")
    assert dispatch({"action": "sections"}, no_client, scope="research") == {
        "sections": ["browser", "requests"]}
    catalog = dispatch({"action": "catalog", "section": "browser"}, no_client, scope="research")
    assert [item["name"] for item in catalog["operations"]] == ["open_page"]
    for action in ("describe", "call"):
        with pytest.raises(PermissionError):
            dispatch({"action": action, "section": "browser", "name": "download", "arguments": {}},
                     no_client, scope="research")
    with pytest.raises(ValueError):
        dispatch({"action": "sections", "scope": "manual"}, no_client, scope="research")
    with pytest.raises(PermissionError):
        dispatch({"action": "sections"}, no_client, scope="unknown")


@pytest.mark.parametrize("address", [("127.0.0.1", 9999), ("localhost", 8000),
                                    ("8.8.8.8", 8000), "/run/session.sock", ("::1", 8000, 0, 0)])
def test_exact_endpoints_refuse_other_services(address):
    policy = EndpointPolicy(["http://127.0.0.1:8000/v1"])
    with pytest.raises(PermissionError):
        policy("socket.connect", (None, address))


def test_exact_endpoints_and_no_datagrams():
    policy = EndpointPolicy(["http://127.0.0.1:8000/v1", "http://[::1]:9000/"])
    policy("socket.connect", (None, ("127.0.0.1", 8000)))
    policy("socket.getaddrinfo", ("::1", 9000, 0, 0))
    for event in ("socket.bind", "socket.sendto", "socket.sendmsg"):
        with pytest.raises(PermissionError):
            policy(event, ())


@pytest.mark.parametrize("url", ["http://localhost:8000", "https://127.0.0.1:8000",
                                "http://8.8.8.8:8000", "http://x:y@127.0.0.1:8000",
                                "http://127.0.0.1:8000/?token=secret"])
def test_invalid_service_configuration(url):
    with pytest.raises(ValueError):
        EndpointPolicy([url])


def test_read_deadline_does_not_reset_per_fragment():
    now = [100.0]
    timeouts = []

    class Connection:
        def settimeout(self, value):
            timeouts.append(value)

    source = io.BytesIO(b"abc")
    reader = DeadlineReader(Connection(), source, clock=lambda: now[0])
    buffer = bytearray(1)
    assert reader.readinto(buffer) == 1
    now[0] += 4
    assert reader.readinto(buffer) == 1
    assert timeouts == [5, 1]
    now[0] += 1
    with pytest.raises(TimeoutError):
        reader.readinto(buffer)
    assert source.tell() == 2
    reader.close()
    assert source.closed
