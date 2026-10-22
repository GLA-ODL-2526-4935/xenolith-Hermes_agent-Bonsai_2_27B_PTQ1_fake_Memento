"""Human/controller commands. Never register these as model tools."""
import json
from pathlib import Path

from .store import Store


def register_cli(subparser):
    commands = subparser.add_subparsers(dest="memory_command", required=True)
    controlled = commands.add_parser("controlled-run", help="Opt-in isolated step executor; read-only selected workspace, no memory layers")
    controlled.add_argument("task_file")
    controlled.add_argument("--session", required=True)
    controlled.add_argument("--workspace", required=True)
    controlled.add_argument("--base-url", required=True)
    controlled.add_argument("--model", required=True)
    controlled.add_argument("--reuse-prefix", action="store_true", help="Opt-in exact-prefix reuse on a dedicated one-slot Prism server")
    controlled.add_argument("--keep-alive", action="store_true", help="Reuse the local HTTP connection; never retry failed requests")
    controlled.add_argument("--idle-handoff-after", type=int, help="Remain foreground and save a separate temporary handoff after N idle seconds")
    commands.add_parser("status", help="Show session ID, busy state and layer counts")
    commands.add_parser("export", help="Print the full human-readable text store")
    commands.add_parser("overview", help="Owner dashboard: records, reviews, tasks and shadow sessions")
    commands.add_parser("backup-records", help="Print portable curated records only; no approvals or credentials")
    commands.add_parser("source-manifest", help="Inspect source SHA256 inventory; does not install or launch")
    restore = commands.add_parser("restore-records", help="Restore curated records into a fresh empty bank only")
    restore.add_argument("archive_file")
    for name, help_text in (("task-state", "Inspect full source-linked task clauses and owner statuses"),
                            ("preview-context", "Preview lossless rebuild; never changes live context"),
                            ("capture-shadow", "Import visible text transcript for offline shadow capture"),
                            ("annotate-task", "Owner declares a task status or explicit supersession")):
        child = commands.add_parser(name, help=help_text)
        child.add_argument("session")
        if name in {"capture-shadow", "annotate-task"}:
            child.add_argument("input_file")
    checkpoint = commands.add_parser("checkpoint", help="Open a five-minute append stage")
    checkpoint.add_argument("session")
    checkpoint.add_argument("evidence_file", help="JSON list of owner-verified evidence")
    edit = commands.add_parser("edit", help="Owner-only replacement from a UTF-8 text file")
    edit.add_argument("id")
    edit.add_argument("text_file")
    delete = commands.add_parser("delete", help="Owner-only permanent record removal")
    delete.add_argument("id")
    commands.add_parser("consolidate", help="Promote records if idle for three hours")
    commands.add_parser("requests", help="Inspect unverified checkpoint candidates")
    commands.add_parser("tasks", help="Inspect separate unfinished-task recovery state")
    step = commands.add_parser("save-step", help="Controller saves visible source-linked recovery, never approved memory")
    step.add_argument("session")
    step.add_argument("step_file", help="Bounded JSON step record; no reasoning/media")
    commands.add_parser("expire", help="Apply the configured retention policy")
    approve = commands.add_parser("approve", help="Owner approves selected evidence IDs")
    approve.add_argument("session")
    approve.add_argument("evidence_ids", nargs="+")
    complete = commands.add_parser("task-complete", help="Owner closes a recovery task")
    complete.add_argument("session")
    sort = commands.add_parser("sort", help="Apply an owner-prepared idle sorting plan")
    sort.add_argument("plan_file", help="JSON object with revision and assignments")
    recover = commands.add_parser("recover", help="Clear abandoned activity markers after a crash")
    recover.add_argument("--hermes-stopped", action="store_true", required=True)
    subparser.set_defaults(func=run)


def run(args):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config_readonly
    backend = load_config_readonly().get("memory", {}).get("e4b_memory", {}).get("backend", "local")
    if backend != "local":
        raise PermissionError("Service memory is owner-managed with e4b_memory.service on the service host; no local fallback")
    if args.memory_command == "controlled-run":
        from .controlled_runtime import run_controlled
        result = run_controlled(args, get_hermes_home())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    store = Store(get_hermes_home() / "e4b_memory")
    result = COMMANDS[args.memory_command](store, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _checkpoint(store, args):
    store.open_checkpoint(args.session, json.loads(Path(args.evidence_file).read_text(encoding="utf-8")))
    return {"checkpoint_open": True, "session": args.session, "expires_in_seconds": 300}


def _edit(store, args):
    store.owner_edit(args.id, content=Path(args.text_file).read_text(encoding="utf-8"))
    return {"edited": args.id}


def _delete(store, args):
    store.owner_edit(args.id, delete=True)
    return {"deleted": args.id, "recoverable": False}


def _recover(store, args):
    with store.transaction() as state:
        state["active"].clear()
        state["stages"].clear()
    return {"recovered": True}


def _status(store, args):
    state = store.snapshot()
    return {"counts": {f"{bucket}.{layer}": sum(r["bucket"] == bucket and r["layer"] == layer for r in state["records"])
                       for bucket in ("temporary", "persistent") for layer in ("L0", "L1", "L2", "L3")},
            "active": state["active"], "checkpoints": list(state["stages"]), "revision": state["revision"],
            "last_activity": state["last_activity"]}


def _approve(store, args):
    store.approve_request(args.session, args.evidence_ids)
    return {"checkpoint_open": True, "session": args.session, "expires_in_seconds": 300}


def _complete(store, args):
    store.complete_task(args.session)
    return {"task_complete": args.session}


def _sort(store, args):
    plan = json.loads(Path(args.plan_file).read_text(encoding="utf-8"))
    return {"promoted": store.apply_sort_plan(plan["revision"], plan["assignments"])}


def _save_step(store, args):
    path = Path(args.step_file)
    if path.stat().st_size > 32000:
        raise ValueError("Recovery input exceeds size limit")
    import io
    from local_extensions.e4b_http import read_json
    raw = path.read_bytes()
    payload = read_json(io.BytesIO(raw), len(raw))
    return store.save_step(args.session, payload)


def _expire(store, args):
    from . import load_settings
    settings = load_settings()
    return {"expired": store.expire(l0_days=settings["l0_days"], recovery_days=settings["recovery_days"])}


COMMANDS = {
    "status": _status, "export": lambda store, args: store.snapshot(),
    "checkpoint": _checkpoint, "edit": _edit, "delete": _delete, "recover": _recover,
    "consolidate": lambda store, args: {"promoted": store.consolidate()},
    "requests": lambda store, args: store.snapshot()["requests"],
    "tasks": lambda store, args: store.snapshot()["tasks"],
    "save-step": _save_step,
    "approve": _approve, "task-complete": _complete, "sort": _sort, "expire": _expire,
}


def _offline_commands():
    from .checkpoint import capture_shadow, preview_rebuild
    from .owner import overview, read_input, record_backup, restore_records, source_manifest
    from .task_state import annotate_task, task_view
    return {"overview": lambda store, args: overview(store),
            "source-manifest": lambda store, args: source_manifest(Path(__file__).parent),
            "backup-records": lambda store, args: record_backup(store),
            "restore-records": lambda store, args: restore_records(store, read_input(args.archive_file, 32 * 1024 * 1024)),
            "task-state": lambda store, args: task_view(store, args.session),
            "preview-context": lambda store, args: preview_rebuild(store, args.session),
            "capture-shadow": lambda store, args: capture_shadow(store, args.session, read_input(args.input_file)),
            "annotate-task": lambda store, args: annotate_task(store, args.session, read_input(args.input_file))}


COMMANDS.update(_offline_commands())
