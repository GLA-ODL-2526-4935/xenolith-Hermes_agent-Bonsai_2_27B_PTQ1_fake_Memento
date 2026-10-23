"""Explicit owner CLI for the service. These commands are never model tools."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .cli import COMMANDS as OWNER_COMMANDS, register_cli
from .service_http import serve
from .service_runtime import Runtime, create
from .sorting import local_endpoint


def settings(value):
    keys = {"semantic_sort", "l0_days", "recovery_days", "base_url", "model"}
    if (not isinstance(value, dict) or not keys <= set(value)
            or set(value) - keys - {"automatic_review", "review_timeout_seconds"}
            or not isinstance(value["semantic_sort"], bool)
            or type(value.get("automatic_review", False)) is not bool):
        raise ValueError("Invalid service settings")
    from .review_model import validate_timeout
    validate_timeout(value.get("review_timeout_seconds", 120))
    for key, maximum in (("l0_days", 3650), ("recovery_days", 365)):
        if type(value[key]) is not int or not 1 <= value[key] <= maximum:
            raise ValueError("Invalid retention interval")
    if value["semantic_sort"] or value.get("automatic_review", False):
        local_endpoint(value["base_url"])
        if not isinstance(value["model"], str) or not 0 < len(value["model"]) <= 200:
            raise ValueError("Explicit same-local-model identifier required")
    return value


def initialize(args):
    create(Path(args.root))
    return {"created": True, "notice": "Issue credentials separately with client; do not mount this root in the agent"}


def issue(args):
    return Runtime(Path(args.root)).credentials.issue()


def revoke(args):
    Runtime(Path(args.root)).credentials.revoke(args.client_id)
    return {"revoked": args.client_id}


def owner(args):
    runtime = Runtime(Path(args.root))
    if args.memory_command == "expire":
        config = settings(json.loads((runtime.root / "settings.json").read_text(encoding="utf-8")))
        return {"expired": runtime.store.expire(l0_days=config["l0_days"], recovery_days=config["recovery_days"])}
    return OWNER_COMMANDS[args.memory_command](runtime.store, args)


def configure(args):
    from .store import file_lock
    runtime = Runtime(Path(args.root))
    value = settings(json.loads(Path(args.file).read_text(encoding="utf-8")))
    # Owner maintenance requires the service stopped; replace settings atomically.
    with file_lock(runtime.root / "service.lock"):
        descriptor, temporary = tempfile.mkstemp(dir=runtime.root, prefix=".settings-", suffix=".json")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, runtime.root / "settings.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {"configured": True}


def run_service(args):
    from .store import file_lock
    runtime = Runtime(Path(args.root))
    settings(json.loads((runtime.root / "settings.json").read_text(encoding="utf-8")))
    with file_lock(runtime.root / "service.lock"):
        serve(runtime, args.port)
    return {"stopped": True}


COMMANDS = {"init": initialize, "client": issue, "revoke": revoke, "owner": owner,
            "configure": configure, "serve": run_service}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Dedicated owner-side service directory")
    commands = parser.add_subparsers(dest="action", required=True)
    for name, handler in COMMANDS.items():
        child = commands.add_parser(name)
        child.set_defaults(service_handler=handler)
        if name == "owner":
            register_cli(child)
        if name == "revoke":
            child.add_argument("client_id")
        if name == "configure":
            child.add_argument("file", help="Validated owner settings JSON; stop service first")
        if name == "serve":
            child.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(args.service_handler(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
