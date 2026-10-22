"""Pure/injected bridge and owner backup contracts."""
import copy

import pytest

from local_extensions.e4b_memory.bridge_contract import prepare_request, scripted_completion
from local_extensions.e4b_memory.owner import overview, read_input, record_backup, restore_records, source_manifest
from local_extensions.e4b_memory.store import Store


def chunk(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def request(tools=False):
    schemas = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}]
    return prepare_request("future-e4b", [{"role": "user", "content": "Inspect the approved item."}], schemas if tools else None)


def test_scripted_text_completion_preserves_request_and_never_dispatches():
    value = request()
    before = copy.deepcopy(value)
    result = scripted_completion(value, lambda req: [chunk({"role": "assistant", "content": "Hello "}),
                                                    chunk({"content": "world"}, "stop")])
    assert result["message"]["content"] == "Hello world"
    assert value == before
    assert result["tools_dispatched"] is False


def test_streamed_tool_fragments_preserve_id_and_arguments():
    values = [chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {
        "name": "inspect", "arguments": '{"item":'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"a"}'}}]}, "tool_calls")]
    result = scripted_completion(request(True), lambda req: values)
    call = result["message"]["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["function"] == {"name": "inspect", "arguments": '{"item":"a"}'}


@pytest.mark.parametrize("chunks", [
    [], [chunk({"content": "partial"})], [chunk({"content": "partial"}, "length")],
    [{"error": {"message": "backend failed"}}],
    [chunk({"role": "user"}, "stop")], [chunk({"reasoning_content": "hidden"}, "stop")],
    [chunk({}, "stop"), chunk({"content": "late"})],
    [chunk({"tool_calls": [{"index": 0, "id": "x", "function": {"name": "unknown", "arguments": "{}"}}]}, "tool_calls")],
    [chunk({"tool_calls": [{"index": 0, "id": "x", "function": {"name": "inspect", "arguments": '{"a":1,"a":2}'}}]}, "tool_calls")],
    [chunk({"tool_calls": [{"index": 0, "id": "x", "function": {"name": "inspect", "arguments": "[]"}}]}, "tool_calls")],
    [chunk({"tool_calls": [{"index": 0, "id": "x", "function": {"name": "inspect", "arguments": "{}"}}]}, "stop")],
])
def test_incomplete_invalid_or_unsupported_responses_never_succeed(chunks):
    with pytest.raises(ValueError):
        scripted_completion(request(True), lambda req: chunks)


def test_cancellation_before_request_and_between_chunks():
    def forbidden(req):
        raise AssertionError("Transport must not start")

    with pytest.raises(InterruptedError):
        scripted_completion(request(), forbidden, lambda: True)
    stopped = False

    def transport(req):
        nonlocal stopped
        yield chunk({"content": "partial"})
        stopped = True
        yield chunk({}, "stop")

    with pytest.raises(InterruptedError):
        scripted_completion(request(), transport, lambda: stopped)


def test_owner_backup_roundtrip_excludes_grants_and_refuses_overwrite(tmp_path):
    source = Store(tmp_path / "source")
    source.open_checkpoint("s", [{"id": "e", "kind": "success", "text": "Owner checked the result."}])
    key = source.append("s", layer="L1", kind="success", label="result", content="Exact value -0.25 V.", evidence_ids=["e"])
    source.activity("s", True)
    archive = record_backup(source)
    assert set(archive) == {"format", "records"}
    target = Store(tmp_path / "target")
    assert restore_records(target, archive)["permissions_restored"] is False
    assert target.snapshot()["records"] == source.snapshot()["records"]
    assert target.snapshot()["stages"] == {} and target.snapshot()["active"] == {}
    assert overview(target)["curated"][0]["id"] == key
    before = target.snapshot()
    with pytest.raises(PermissionError):
        restore_records(target, archive)
    assert target.snapshot() == before


def test_invalid_backup_is_atomic_and_duplicate_owner_json_refused(tmp_path):
    store = Store(tmp_path / "bank")
    before = store.snapshot()
    with pytest.raises(ValueError):
        restore_records(store, {"format": "e4b-curated-records-v1", "records": [], "stages": {"s": "grant"}})
    assert store.snapshot() == before
    path = tmp_path / "input.json"
    path.write_text('{"id":1,"id":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_input(path)


def test_source_manifest_changes_with_content_without_installing(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("First package description.", encoding="utf-8")
    before = source_manifest(tmp_path)
    path.write_text("Updated package description.", encoding="utf-8")
    after = source_manifest(tmp_path)
    assert before["files"]["README.md"] != after["files"]["README.md"]
    assert after["runtime_certified"] is False
