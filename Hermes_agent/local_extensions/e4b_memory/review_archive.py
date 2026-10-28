"""Owner-only revision-bound archival; durable full audit before queue removal."""
import copy
import json
import os
import re
import time
import uuid

from .review_display import disposition
from .step_recovery import digest


def _eligible(state, ids):
    if (not isinstance(ids, list) or not ids or len(ids) > 128
            or any(not isinstance(key, str) for key in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Select 1–128 distinct proposal IDs")
    proposals = state.get("memory_proposals", {})
    selected = {}
    for key in ids:
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{32}", key) or key not in proposals:
            raise ValueError("Unknown proposal ID")
        proposal = proposals[key]
        if disposition(state, proposal) == "pending":
            raise PermissionError("Decide current pending proposals before archiving")
        selected[key] = copy.deepcopy(proposal)
    return selected


def preview_archive(store, ids):
    state = store.snapshot()
    selected = _eligible(state, ids)
    return {"format": "memory-archive-plan-v1", "revision": state["revision"],
            "ids": sorted(selected), "digest": digest(selected), "count": len(selected),
            "effect": "Back up full selected audits, then remove them from the live queue. Memory records are unchanged."}


def _save_archive(root, payload):
    folder = root / "proposal-archives"
    if folder.is_symlink():
        raise PermissionError("Archive directory must not be a symlink")
    folder.mkdir(mode=0o700, exist_ok=True)
    path = folder / (uuid.uuid4().hex + ".json")
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != raw:
        raise OSError("Archive readback failed; live proposals retained")
    if os.name != "nt":
        directory = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return path


def archive(store, plan):
    expected = {"format", "revision", "ids", "digest", "count", "effect"}
    if (not isinstance(plan, dict) or set(plan) != expected or plan["format"] != "memory-archive-plan-v1"
            or type(plan["revision"]) is not int or type(plan["count"]) is not int):
        raise ValueError("Use an unmodified archive preview plan")
    with store.transaction() as state:
        if state["active"] or state["revision"] != plan["revision"]:
            raise PermissionError("Busy or changed bank; preview again")
        selected = _eligible(state, plan["ids"])
        if len(selected) != plan["count"] or digest(selected) != plan["digest"]:
            raise PermissionError("Archive selection changed")
        archived = state.setdefault("memory_review_archived_bases", {})
        bases = {p["job"]["basis"] for p in selected.values()}
        if len(set(archived) | bases) > 4096:
            raise ValueError("Archived source-signature capacity reached; owner maintenance required")
        path = _save_archive(store.root, {"format": "memory-proposal-archive-v1", "created": time.time(),
                                          "source_revision": state["revision"], "proposals": selected})
        for key, proposal in selected.items():
            archived[proposal["job"]["basis"]] = disposition(state, proposal)
            state["memory_proposals"].pop(key)
        remaining = {p["job"]["basis"] for p in state.get("memory_proposals", {}).values()}
        for basis in bases - remaining:
            state.get("memory_review_attempts", {}).pop(basis, None)
    current = store.snapshot()
    if any(key in current.get("memory_proposals", {}) for key in selected):
        raise OSError("Archive commit unconfirmed; inspect live state before repeating")
    return {"archived": len(selected), "backup_file": str(path), "memory_records_changed": False}


def list_archives(store):
    folder = store.root / "proposal-archives"
    if folder.is_symlink():
        raise PermissionError("Archive directory must not be a symlink")
    files = []
    if folder.exists():
        for path in sorted(folder.iterdir()):
            if re.fullmatch(r"[a-f0-9]{32}\.json", path.name) and path.is_file() and not path.is_symlink():
                files.append({"file": path.name, "bytes": path.stat().st_size})
    return {"files": files, "notice": "Private owner audit backups; not pending approvals or model-readable memories"}


def review_archived(store, filename, *, format="markdown"):
    from .owner import read_input
    from .review_display import proposal_markdown
    from .store import MAX_BYTES
    if not isinstance(filename, str) or not re.fullmatch(r"[a-f0-9]{32}\.json", filename):
        raise ValueError("Select an archive filename from memory-archives")
    folder = store.root / "proposal-archives"
    path = folder / filename
    if folder.is_symlink() or path.is_symlink():
        raise PermissionError("Linked archive files are not supported")
    value = read_input(path, MAX_BYTES)
    if value.get("format") != "memory-proposal-archive-v1" or not isinstance(value.get("proposals"), dict):
        raise ValueError("Not a proposal archive")
    if format == "json":
        return value
    if format != "markdown":
        raise ValueError("Unsupported archive view format")
    return "# Archived audit — read-only, not actionable\n\n" + "\n".join(
        proposal_markdown(p) for p in value["proposals"].values())
