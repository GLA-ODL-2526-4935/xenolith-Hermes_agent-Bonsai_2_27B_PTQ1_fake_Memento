"""Owner-only restart-applied switch; never grants automatic acceptance."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .store import file_lock


def set_workflow(path, enabled, *, kind):
    if type(enabled) is not bool or kind not in {"local", "service"}:
        raise ValueError("Explicit boolean and config kind required")
    path = Path(path).resolve(strict=True)
    with file_lock(path.parent / "memory-workflow-config.lock"):
        original = path.read_text(encoding="utf-8")
        from .owner import read_input
        config = read_input(path)
        if kind == "service":
            from .service import settings
            config["automatic_review"] = enabled
            settings(config)
        else:
            from .restricted import profile_config
            model = config["model"]
            memory = config["memory"]["e4b_memory"]
            memory["automatic_review"] = enabled
            expected = profile_config(model["base_url"], model["default"], semantic_sort=memory.get("semantic_sort", False),
                                      shadow_capture=memory.get("shadow_capture", False),
                                      research_validator=config.get("research_only", {}).get("validator_enabled", True),
                                      memory_service_url=memory.get("service_url"), automatic_review=enabled)
            if not enabled:
                memory.pop("automatic_review", None)
            if "research_only" not in config:
                expected.pop("research_only")
            if config != expected:
                raise PermissionError("Switch supports only the generated restricted local profile")
        payload = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".workflow-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.read_text(encoding="utf-8") != original:
                raise PermissionError("Configuration changed; retry from latest version")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if path.read_text(encoding="utf-8") != payload:
        raise OSError("Switch readback unconfirmed")
    return {"automatic_generation_and_conflict_checks": enabled, "acceptance": "manual", "restart_required": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("state", choices=("on", "off"))
    parser.add_argument("--kind", choices=("local", "service"), required=True)
    parser.add_argument("--stopped", required=True, action="store_true", help="Confirm owning Hermes/service is stopped")
    args = parser.parse_args(argv)
    print(json.dumps(set_workflow(args.config, args.state == "on", kind=args.kind), indent=2))


if __name__ == "__main__":
    main()
