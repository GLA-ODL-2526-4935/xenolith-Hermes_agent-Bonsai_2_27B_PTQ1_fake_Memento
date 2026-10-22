"""Strict parsing regressions for both services; real handlers, no sockets."""
import io
import json
from types import SimpleNamespace

import pytest

from local_extensions.e4b_access.broker import Broker
from local_extensions.e4b_access.service import handler_for as access_handler
from local_extensions.e4b_http import read_json
from local_extensions.e4b_memory.service_http import handler_for as memory_handler
from local_extensions.e4b_memory.service_runtime import create
from tests.local_extensions.test_e4b_access import FakeSocket
from tests.local_extensions.test_e4b_memory_service import begin


@pytest.fixture(params=["memory", "access"])
def endpoint(request, tmp_path):
    if request.param == "access":
        broker, token = Broker.create(tmp_path / "broker")
        return access_handler(broker), token, "/v1/requests", {
            "action": "submit", "proposal": {"kind": "text_export", "label": "Report", "text": "Plain report"}}
    runtime = create(tmp_path / "bank")
    tokens = runtime.credentials.issue()
    begin(runtime, tokens)
    return memory_handler(runtime), tokens["model"], "/v1/memory", {
        "session": "session", "instance": "instance", "action": "memory", "arguments": {"action": "sections"}}


def send(endpoint, *, extra="", payload=None, declared_length=None):
    handler, token, path, body = endpoint
    payload = json.dumps(body).encode() if payload is None else payload
    length = str(len(payload)) if declared_length is None else declared_length
    head = (f"POST {path} HTTP/1.0\r\nHost: 127.0.0.1:8767\r\nContent-Type: application/json\r\n"
            f"Authorization: Bearer {token}\r\nContent-Length: {length}\r\n{extra}\r\n")
    connection = FakeSocket(head.encode() + payload)
    handler(connection, ("127.0.0.1", 1), SimpleNamespace(server_port=8767))
    return bytes(connection.output)


@pytest.mark.parametrize("header", ["Authorization: Bearer wrong\r\n", "aUtHoRiZaTiOn: Bearer wrong\r\n",
                                     "Host: evil.example\r\n", "Content-Type: text/plain\r\n",
                                     "Origin:\r\n", "Transfer-Encoding:\r\n"])
def test_sensitive_duplicate_or_empty_forbidden_header_refused(endpoint, header):
    assert b"400 Bad Request" in send(endpoint, extra=header)


@pytest.mark.parametrize("length", ["+100", "-1", "1_0", "0", "9999999"])
def test_noncanonical_or_out_of_range_lengths_refused(endpoint, length):
    assert b"400 Bad Request" in send(endpoint, declared_length=length)


def test_truncated_request_refused(endpoint):
    payload = json.dumps(endpoint[3]).encode()
    assert b"400 Bad Request" in send(endpoint, payload=payload, declared_length=str(len(payload) + 1))


def test_duplicate_nested_key_refused_even_with_identical_values(endpoint):
    payload = json.dumps(endpoint[3]).replace('"action":', '"action":"ignored","action":', 1).encode()
    assert b"400 Bad Request" in send(endpoint, payload=payload)


def test_normal_request_still_works_after_refusal(endpoint):
    assert b"400 Bad Request" in send(endpoint, extra="Host: evil.example\r\n")
    response = send(endpoint)
    assert b"200 OK" in response
    assert json.loads(response.split(b"\r\n\r\n", 1)[1])["success"] is True


@pytest.mark.parametrize("payload", [b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}',
                                      b'{"outer":{"key":1,"key":2}}', b"[" * 65 + b"0" + b"]" * 65])
def test_nonstandard_or_ambiguous_nested_json_refused(payload):
    with pytest.raises(ValueError):
        read_json(io.BytesIO(payload), len(payload))


def test_unicode_and_quoted_brackets_do_not_trigger_depth_limit():
    body = {"text": '中文 — ' + '[{\\"quoted' * 100, "nested": [{"value": "normal"}]}
    payload = json.dumps(body, ensure_ascii=False).encode()
    assert read_json(io.BytesIO(payload), len(payload)) == body
