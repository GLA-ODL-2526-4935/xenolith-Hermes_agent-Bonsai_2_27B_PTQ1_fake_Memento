"""Offline adversarial contracts: disposable banks and in-memory HTTP only."""
import json
from types import SimpleNamespace

import pytest

from local_extensions.e4b_memory.service_http import handler_for
from local_extensions.e4b_memory.service_runtime import create
from tests.local_extensions.test_e4b_access import FakeSocket
from tests.local_extensions.test_e4b_memory_service import begin, call, describe, memory


@pytest.fixture
def service(tmp_path):
    runtime = create(tmp_path / "bank")
    tokens = runtime.credentials.issue()
    begin(runtime, tokens)
    return runtime, tokens


def request(runtime, token, *, body=None, extra="", path="/v1/memory", host="127.0.0.1:8767"):
    if body is None:
        body = json.dumps({"session": "session", "instance": "instance", "action": "memory",
                           "arguments": {"action": "sections"}}).encode()
    head = (f"POST {path} HTTP/1.0\r\nHost: {host}\r\nContent-Type: application/json\r\n"
            f"Authorization: Bearer {token}\r\nContent-Length: {len(body)}\r\n{extra}\r\n")
    connection = FakeSocket(head.encode() + body)
    handler_for(runtime)(connection, ("127.0.0.1", 1), SimpleNamespace(server_port=8767))
    return bytes(connection.output)


@pytest.mark.parametrize("extra", ["Origin: https://evil.example\r\n", "Transfer-Encoding: chunked\r\n",
                                     "Content-Length: 1\r\n"])
def test_http_framing_attacks_leave_bank_unchanged(service, extra):
    runtime, tokens = service
    before = runtime.store.snapshot()
    response = request(runtime, tokens["model"], extra=extra)
    assert b"400 Bad Request" in response
    assert runtime.store.snapshot() == before


@pytest.mark.parametrize("path", ["/approve", "/v1/memory/../owner", "/v1/memory?admin=true", "/v1/%6demory"])
def test_route_confusion_refused(service, path):
    runtime, tokens = service
    assert b"400 Bad Request" in request(runtime, tokens["model"], path=path)


@pytest.mark.parametrize("host", ["evil.example", "localhost:8767", "127.0.0.1:8767.evil.example", "[::1]:8767"])
def test_host_confusion_refused(service, host):
    runtime, tokens = service
    assert b"400 Bad Request" in request(runtime, tokens["model"], host=host)


@pytest.mark.parametrize("body", [b"{", b"null", b"[]", b"true", b'"text"', b"\xff", b"x" * 32001])
def test_invalid_and_oversized_json_refused(service, body):
    runtime, tokens = service
    before = runtime.store.snapshot()
    assert b"400 Bad Request" in request(runtime, tokens["model"], body=body)
    assert runtime.store.snapshot() == before


def test_unauthorized_request_not_parsed_or_credential_echoed(service):
    runtime, tokens = service
    response = request(runtime, "invalid-credential", body=b"[" * 2000)
    assert b"400 Bad Request" in response
    assert tokens["model"].encode() not in response and b"invalid-credential" not in response


@pytest.mark.parametrize("field", ["session", "instance"])
@pytest.mark.parametrize("value", ["../owner", "a:b", "", "x" * 101, ["session"]])
def test_identifier_injection_refused_without_state_change(service, field, value):
    runtime, tokens = service
    body = {"session": "session", "instance": "instance", "action": "memory", "arguments": {"action": "sections"}}
    body[field] = value
    before = runtime.store.snapshot()
    with pytest.raises((ValueError, PermissionError)):
        runtime.dispatch(tokens["model"], body)
    assert runtime.store.snapshot() == before


def test_client_and_instance_cannot_borrow_another_surface(service):
    runtime, tokens = service
    other = runtime.credentials.issue()
    with pytest.raises(PermissionError):
        memory(runtime, other, {"action": "sections"})
    with pytest.raises(PermissionError):
        call(runtime, tokens, "memory", {"action": "sections"}, role="model", instance="other")
    begin(runtime, other)
    call(runtime, tokens, "capture", {"candidates": [], "error": "owner-client-one-error"})
    describe(runtime, other, "recovery", "recovery")
    response = memory(runtime, other, {"action": "call", "section": "recovery", "name": "recovery"})
    assert "owner-client-one-error" not in json.dumps(response)


def append_args():
    return {"action": "call", "section": "checkpoint", "name": "append", "arguments": {
        "layer": "L1", "kind": "success", "label": "Result", "content": "Finished", "evidence_ids": ["proof"]}}


def test_expired_and_cross_session_append_grants_refused(service):
    runtime, tokens = service
    describe(runtime, tokens, "checkpoint", "append")
    proof = [{"id": "proof", "kind": "success", "text": "Owner checked"}]
    runtime.store.open_checkpoint(tokens["client_id"] + ":session", proof, now=0, ttl=1)
    with pytest.raises(PermissionError):
        memory(runtime, tokens, append_args())
    runtime.store.open_checkpoint(tokens["client_id"] + ":other-session", proof)
    with pytest.raises(PermissionError):
        memory(runtime, tokens, append_args())
    assert runtime.store.snapshot()["records"] == []


def test_append_grant_exhaustion_and_wrong_evidence_kind(service):
    runtime, tokens = service
    describe(runtime, tokens, "checkpoint", "append")
    runtime.store.open_checkpoint(tokens["client_id"] + ":session",
                                  [{"id": "proof", "kind": "success", "text": "Owner checked"}], budget=1)
    args = append_args()
    args["arguments"]["kind"] = "critical_failure"
    with pytest.raises(PermissionError):
        memory(runtime, tokens, args)
    args["arguments"]["kind"] = "success"
    memory(runtime, tokens, args)
    args["arguments"]["content"] = "A different claim"
    with pytest.raises(PermissionError):
        memory(runtime, tokens, args)
    assert len(runtime.store.snapshot()["records"]) == 1


def test_model_cannot_reset_retrieval_budget(service):
    runtime, tokens = service
    session = tokens["client_id"] + ":session"
    runtime.store.open_checkpoint(session, [{"id": "proof", "kind": "success", "text": "Owner checked"}])
    record = runtime.store.append(session, **append_args()["arguments"])
    describe(runtime, tokens, "temporary.L1", "read")
    args = {"action": "call", "section": "temporary.L1", "name": "read", "arguments": {"id": record}}
    for _ in range(3):
        memory(runtime, tokens, args)
    for action in ("begin", "finish", "compress", "close"):
        with pytest.raises(PermissionError):
            call(runtime, tokens, action, role="model")
    describe(runtime, tokens, "temporary.L1", "read")
    with pytest.raises(ValueError, match="budget"):
        memory(runtime, tokens, args)


def test_revoked_tokens_cannot_use_existing_surface(service):
    runtime, tokens = service
    runtime.credentials.revoke(tokens["client_id"])
    before = runtime.store.snapshot()
    for role in ("model", "controller"):
        assert b"400 Bad Request" in request(runtime, tokens[role])
    assert runtime.store.snapshot() == before


def test_nested_json_returns_controlled_refusal(service):
    runtime, tokens = service
    response = request(runtime, tokens["model"], body=b"[" * 2000 + b"0" + b"]" * 2000)
    assert b"400 Bad Request" in response


def test_duplicate_authorization_is_rejected(service):
    runtime, tokens = service
    response = request(runtime, tokens["model"], extra="Authorization: Bearer invalid-credential\r\n")
    assert b"400 Bad Request" in response


def test_duplicate_json_keys_are_rejected(service):
    runtime, tokens = service
    body = (b'{"session":"other","session":"session","instance":"instance",'
            b'"action":"memory","arguments":{"action":"sections"}}')
    assert b"400 Bad Request" in request(runtime, tokens["model"], body=body)
