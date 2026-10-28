"""Owner-side coordinated Linux start. Dry-run default; no install/download fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


FIELDS = {"schema_version", "python_executable", "prism_executable", "prism_sha256", "model_path",
          "hermes_profile", "engine_profile", "automatic_memory_workflow", "execution_enabled",
          "target_hardware_reviewed", "runtime_reviewed", "startup_timeout_seconds"}


def read_config(path):
    from .owner import read_input
    value = read_input(path)
    if not isinstance(value, dict) or set(value) != FIELDS or value["schema_version"] != 1:
        raise ValueError("Use the complete launch.example.json schema")
    for key in ("automatic_memory_workflow", "execution_enabled", "target_hardware_reviewed", "runtime_reviewed"):
        if type(value[key]) is not bool:
            raise ValueError("Expected boolean " + key)
    for key in ("python_executable", "prism_executable", "model_path", "hermes_profile"):
        item = value[key]
        if not isinstance(item, str) or (item and (not Path(item).is_absolute() or any(c in item for c in "\x00\r\n"))):
            raise ValueError("Use an absolute configured path or blank: " + key)
    if not isinstance(value["prism_sha256"], str) or (value["prism_sha256"] and
            (len(value["prism_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in value["prism_sha256"]))):
        raise ValueError("Runtime digest must be blank or SHA256")
    if type(value["startup_timeout_seconds"]) is not int or not 10 <= value["startup_timeout_seconds"] <= 3600:
        raise ValueError("Startup timeout must be 10–3600 seconds")
    if value["engine_profile"] not in {"bonsai27-linux-cpu.json", "bonsai27-linux-vulkan.json"}:
        raise ValueError("Choose an existing baseline CPU or Vulkan profile; no optimization auto-activation")
    return value


def build_plan(config, repo):
    """Read source configuration only; never open model/runtime or contact endpoints."""
    from engine.backends.prism import load_profile, make_plan
    repo = Path(repo).resolve()
    engine_profile = repo / "engine" / "profiles" / config["engine_profile"]
    engine = load_profile(engine_profile)
    blocked = ["Fill " + k for k in ("python_executable", "prism_executable", "prism_sha256", "model_path", "hermes_profile")
               if not config[k]]
    blocked += ["Review/enable " + k for k in ("execution_enabled", "target_hardware_reviewed", "runtime_reviewed")
                if not config[k]]
    plan = {"mode": "dry-run", "blocked": blocked, "endpoint": f"http://127.0.0.1:{engine['port']}/v1",
            "model": "bonsai-2-27b", "engine_argv": None, "hermes_argv": None,
            "automatic_memory_workflow": config["automatic_memory_workflow"], "acceptance": "manual",
            "profile": config["hermes_profile"], "timeout": config["startup_timeout_seconds"],
            "prism_sha256": config["prism_sha256"], "browser": "not connected; memory-only Hermes"}
    if config["prism_executable"] and config["model_path"]:
        plan["engine_plan"] = make_plan(engine, config["prism_executable"], config["model_path"])
    if not blocked:
        python = config["python_executable"]
        plan["engine_argv"] = [python, str(repo / "engine" / "launch.py"), "--profile", str(engine_profile),
                               "--engine", config["prism_executable"], "--model", config["model_path"], "--start"]
        plan["hermes_argv"] = [python, "-m", "local_extensions.e4b_memory.restricted", "run", config["hermes_profile"]]
    return plan


def host_blockers(facts):
    """Pure compatibility screening of observed facts, not a hardware certification."""
    reasons = []
    if facts["os"] != "linux" or facts["arch"] != "amd64":
        reasons.append("Native Linux x86-64 required; Windows/macOS/ARM are preparation-only")
    if facts["wsl"] or facts["container"]:
        reasons.append("This launcher requires the reviewed native target, not WSL/container")
    return reasons


def current_host():
    from hermes_platform.host import facts, runtime
    return {"os": facts.os_family(), "arch": facts.native_arch(), "wsl": runtime.is_wsl(),
            "container": runtime.is_container()}


def prepare_profile(plan):
    from .restricted import create_profile, profile_config
    from .workflow_switch import set_workflow
    from .owner import source_manifest
    root = Path(plan["profile"])
    if not root.exists():
        create_profile(root, plan["endpoint"], plan["model"], automatic_review=plan["automatic_memory_workflow"])
        return
    actual = json.loads((root / "config.yaml").read_text(encoding="utf-8"))
    enabled = actual.get("memory", {}).get("e4b_memory", {}).get("automatic_review", False)
    expected = profile_config(plan["endpoint"], plan["model"], automatic_review=enabled)
    if actual != expected:
        raise PermissionError("Existing profile differs from generated restricted settings; refusing overwrite")
    source = Path(__file__).parent
    if source_manifest(source)["files"] != source_manifest(root / "plugins" / "e4b_memory")["files"]:
        raise PermissionError("Profile plugin copy differs; explicitly back up/refresh it before starting")
    if enabled != plan["automatic_memory_workflow"]:
        set_workflow(root / "config.yaml", plan["automatic_memory_workflow"], kind="local")


def validate_runtime(plan):
    python, executable = Path(plan["engine_argv"][0]), Path(plan["engine_plan"]["argv"][0])
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Owner-configured Hermes Python is missing/not executable")
    with executable.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != plan["prism_sha256"]:
        raise ValueError("Prism executable digest mismatch; no process started")


def desktop_file(repo, config_path, destination):
    """Create an owner-requested KDE entry; never overwrite/install/autostart it."""
    def quoted(value):
        value = str(value)
        if any(c in value for c in "\x00\r\n%"):
            raise ValueError("Unsupported desktop launcher path")
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$") + '"'
    command = " ".join(quoted(v) for v in (sys.executable, Path(repo) / "start.py", "--config", config_path, "--start"))
    text = "[Desktop Entry]\nType=Application\nName=Start local Hermes experiment\nTerminal=true\nExec=" + command + "\n"
    with Path(destination).open("x", encoding="utf-8") as handle:
        handle.write(text)
    return {"created": str(destination), "notice": "Review/trust this entry on KDE; no process started or autostart installed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("launch.local.json"))
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--desktop", type=Path, help="Create a new KDE launcher file only; never starts")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    if args.start and args.desktop:
        parser.error("Create a launcher or start; not both")
    try:
        if args.desktop:
            result = desktop_file(repo, args.config.resolve(), args.desktop)
        else:
            config = read_config(args.config)
            plan = build_plan(config, repo)
            if args.start:
                blockers = plan["blocked"] + host_blockers(current_host())
                if blockers:
                    raise PermissionError("Start blocked: " + "; ".join(blockers))
                from .launch_runtime import run_stack
                return run_stack(plan, repo)
            result = plan
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, PermissionError, KeyError, RuntimeError) as exc:
        parser.exit(2, str(exc) + "\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
