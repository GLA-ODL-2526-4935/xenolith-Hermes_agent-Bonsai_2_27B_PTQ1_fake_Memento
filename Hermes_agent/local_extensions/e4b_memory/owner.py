"""Owner-only inspection and portable curated-record backups; no credentials/grants."""
import copy
import io
import hashlib
import json
from pathlib import Path

from .store import MAX_BYTES, validate_record


def source_manifest(root):
    root = Path(root).resolve(strict=True)
    files = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix not in {".py", ".md", ".yaml", ".json", ".toml"}:
            continue
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError("Package manifest refuses linked source files")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"format": "e4b-source-sha256-v1", "files": files,
            "dependency_authority": "Hermes pyproject.toml and uv.lock; no added dependencies",
            "runtime_certified": False}


def read_input(path, maximum=128000):
    from local_extensions.e4b_http import read_json
    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError("Owner input exceeds size budget")
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise ValueError("Owner input exceeds size budget")
    return read_json(io.BytesIO(raw), len(raw))


def overview(store):
    state = store.snapshot()
    return {"revision": state["revision"], "busy": bool(state["active"]),
            "curated": [{"id": r["id"], "layer": r["layer"], "bucket": r["bucket"],
                         "kind": r["kind"], "label": r["label"]} for r in state["records"]],
            "pending_reviews": list(state["requests"]), "tasks": state["tasks"],
            "shadow_sessions": [{"session": s, "revision": c["revision"],
                                 "source_count": len(c["references"]), "shadow_only": True}
                                for s, c in state.get("shadow_checkpoints", {}).items()]}


def record_backup(store):
    """Recovery/transcripts deliberately excluded; use owner export for full inspection."""
    return {"format": "e4b-curated-records-v1", "records": copy.deepcopy(store.snapshot()["records"])}


def restore_records(store, archive):
    if (not isinstance(archive, dict) or set(archive) != {"format", "records"}
            or archive["format"] != "e4b-curated-records-v1" or not isinstance(archive["records"], list)):
        raise ValueError("Expected a curated-records backup, not a live store or credential archive")
    if len(json.dumps(archive, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
        raise ValueError("Backup exceeds store budget")
    seen = set()
    for record in archive["records"]:
        validate_record(record)
        key = record.get("id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Missing/duplicate restored record ID")
        seen.add(key)
    with store.transaction() as state:
        if (state["revision"] != 0 or state["records"] or state["active"] or state["stages"]
                or state.get("step_recovery") or state.get("shadow_checkpoints")):
            raise PermissionError("Restore only into a fresh empty bank; existing data is never overwritten")
        state["records"] = copy.deepcopy(archive["records"])
    if store.snapshot()["records"] != archive["records"]:
        raise OSError("Restore readback failed")
    return {"restored_records": len(seen), "permissions_restored": False, "recovery_restored": False}
