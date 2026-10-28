"""Offline launch plans and scripted lifetime checks; no model/engine subprocess."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_extensions.e4b_memory.launch_stack import (build_plan, current_host, desktop_file, host_blockers,
                                                     main, prepare_profile, read_config, validate_runtime)
from local_extensions.e4b_memory.launch_runtime import child_environment, free_endpoint, supervise


@pytest.fixture
def config(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo))
    config = {"schema_version": 1, "python_executable": str(tmp_path / "python"),
              "prism_executable": str(tmp_path / "llama-server"), "prism_sha256": "a" * 64,
              "model_path": str(tmp_path / "Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf"),
              "hermes_profile": str(tmp_path / "profile"), "engine_profile": "bonsai27-linux-cpu.json",
              "automatic_memory_workflow": False, "execution_enabled": True,
              "target_hardware_reviewed": True, "runtime_reviewed": True, "startup_timeout_seconds": 600}
    return config, repo


def test_complete_dry_plan_does_not_need_model_runtime_or_python_files(config):
    value, repo = config
    plan = build_plan(value, repo)
    assert not plan["blocked"]
    assert plan["endpoint"] == plan["engine_plan"]["endpoint"]
    assert plan["model"] == plan["engine_plan"]["request_defaults"]["model"]
    assert plan["hermes_argv"][-1] == value["hermes_profile"]
    assert plan["engine_argv"][-1] == "--start"
    assert plan["engine_plan"]["argv"][plan["engine_plan"]["argv"].index("--device") + 1] == "none"
    assert not Path(value["hermes_profile"]).exists()


def test_blank_paths_and_off_switches_are_blockers_not_downloads(config):
    value, repo = config
    for name in ("python_executable", "prism_executable", "prism_sha256", "model_path", "hermes_profile"):
        value[name] = ""
    value.update(execution_enabled=False, target_hardware_reviewed=False, runtime_reviewed=False)
    plan = build_plan(value, repo)
    assert plan["blocked"] and plan["engine_argv"] is None and plan["hermes_argv"] is None
    assert "engine_plan" not in plan


@pytest.mark.parametrize("facts", [
    {"os": "win32", "arch": "amd64", "wsl": False, "container": False},
    {"os": "darwin", "arch": "arm64", "wsl": False, "container": False},
    {"os": "linux", "arch": "arm64", "wsl": False, "container": False},
    {"os": "linux", "arch": "amd64", "wsl": True, "container": False},
    {"os": "linux", "arch": "amd64", "wsl": False, "container": True},
])
def test_host_screening_is_pure_and_refuses_unsupported_targets(facts):
    assert host_blockers(facts)


def test_host_api_uses_real_native_facts_without_running_gpu():
    observed = current_host()
    assert isinstance(observed["os"], str) and isinstance(observed["arch"], str)
    assert type(observed["wsl"]) is bool and type(observed["container"]) is bool
    assert host_blockers({"os": "linux", "arch": "amd64", "wsl": False, "container": False}) == []


@pytest.mark.parametrize("field,value", [("automatic_memory_workflow", "yes"), ("execution_enabled", 1),
                                       ("engine_profile", "../../evil.json"), ("prism_sha256", "not-a-hash"),
                                       ("startup_timeout_seconds", True), ("model_path", "relative.gguf")])
def test_configuration_rejects_ambiguous_or_arbitrary_launch_choices(config, tmp_path, field, value):
    source, _ = config
    source[field] = value
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        read_config(path)


def test_profile_endpoint_is_wired_and_workflow_switch_is_restart_applied(config):
    value, repo = config
    plan = build_plan(value, repo)
    prepare_profile(plan)
    path = Path(plan["profile"]) / "config.yaml"
    original = json.loads(path.read_text())
    assert original["model"]["base_url"] == plan["endpoint"]
    assert original["model"]["default"] == plan["model"]
    assert original["platform_toolsets"] == {"cli": ["memory"]}
    plan["automatic_memory_workflow"] = True
    prepare_profile(plan)
    actual = json.loads(path.read_text())
    assert actual["memory"]["e4b_memory"]["automatic_review"] is True
    assert actual["model"] == original["model"] and actual["platform_toolsets"] == original["platform_toolsets"]


def test_launcher_never_overwrites_a_changed_profile(config):
    value, repo = config
    plan = build_plan(value, repo)
    prepare_profile(plan)
    path = Path(plan["profile"]) / "config.yaml"
    altered = json.loads(path.read_text())
    altered["platform_toolsets"] = {"cli": ["terminal"]}
    path.write_text(json.dumps(altered))
    with pytest.raises(PermissionError):
        prepare_profile(plan)
    assert json.loads(path.read_text()) == altered


def test_launcher_refuses_stale_plugin_copy(config):
    value, repo = config
    plan = build_plan(value, repo)
    prepare_profile(plan)
    # This edits disposable runtime data, not reads source code as a shape test.
    copied = Path(plan["profile"]) / "plugins" / "e4b_memory" / "owner-note.md"
    copied.write_text("Different owner plugin snapshot")
    with pytest.raises(PermissionError):
        prepare_profile(plan)


def test_child_environment_does_not_leak_owner_token_or_hosted_fallback(config, monkeypatch):
    monkeypatch.setenv("DISCORD_MEMORY_REVIEW_TOKEN", "synthetic")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("PYTHONPATH", "untrusted")
    env = child_environment(config[0]["hermes_profile"])
    assert "DISCORD_MEMORY_REVIEW_TOKEN" not in env and "OPENAI_API_KEY" not in env and "PYTHONPATH" not in env
    assert env["HERMES_HOME"] == config[0]["hermes_profile"]


def test_runtime_pin_failure_occurs_before_any_process(config, tmp_path):
    value, repo = config
    value["python_executable"] = __import__("sys").executable
    Path(value["prism_executable"]).write_bytes(b"not a model runtime")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_runtime(build_plan(value, repo))


@pytest.mark.parametrize("url", ["https://127.0.0.1:18780/v1", "http://example.com:18780/v1", "http://127.0.0.1/v1"])
def test_readiness_cannot_use_external_or_implicit_endpoint(url):
    with pytest.raises(ValueError):
        free_endpoint(url)


def test_desktop_entry_is_explicit_new_file_only_and_does_not_execute(config, tmp_path):
    _, repo = config
    entry = tmp_path / "Start Project.desktop"
    desktop_file(repo, tmp_path / "launch.local.json", entry)
    text = entry.read_text()
    assert "Terminal=true" in text and "--start" in text and "launch.local.json" in text
    with pytest.raises(FileExistsError):
        desktop_file(repo, tmp_path / "launch.local.json", entry)


def test_default_main_is_dry_run(config, tmp_path, capsys):
    value, _ = config
    path = tmp_path / "launch.local.json"
    path.write_text(json.dumps(value))
    assert main(["--config", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run" and not Path(value["hermes_profile"]).exists()


def test_scripted_lifecycle_waits_before_hermes_and_cleans_owned_children():
    events = []
    server = SimpleNamespace(poll=lambda: None)
    agent = SimpleNamespace(poll=lambda: 0, returncode=0)
    def spawn(argv, **kwargs):
        assert "shell" not in kwargs
        events.append(argv[0])
        return server if argv[0] == "engine" else agent
    def ready(process, endpoint, timeout):
        assert process is server
        events.append("ready")
    engine = {"argv": ["engine"], "cwd": "/unused", "env": {}, "endpoint": "http://127.0.0.1:9999/v1", "timeout": 10}
    hermes = {"argv": ["hermes"], "cwd": "/unused", "env": {}}
    stopped = []
    assert supervise(engine, hermes, spawn=spawn, wait=ready, stop=stopped.append) == 0
    assert events == ["engine", "ready", "hermes"] and stopped == [agent, server]


def test_scripted_readiness_failure_never_starts_hermes():
    engine = {"argv": ["engine"], "cwd": "/unused", "env": {}, "endpoint": "http://127.0.0.1:9999/v1", "timeout": 10}
    spawns, stopped = [], []
    server = object()
    def spawn(argv, **_):
        spawns.append(argv)
        return server
    def fail(*_):
        raise RuntimeError("not ready")
    with pytest.raises(RuntimeError):
        supervise(engine, copy.deepcopy(engine), spawn=spawn, wait=fail, stop=stopped.append)
    assert spawns == [["engine"]] and stopped == [server]


def test_scripted_agent_spawn_failure_stops_owned_engine():
    engine = {"argv": ["engine"], "cwd": "/unused", "env": {}, "endpoint": "http://127.0.0.1:9999/v1", "timeout": 10}
    server, stopped = object(), []
    def spawn(argv, **_):
        if argv == ["engine"]:
            return server
        raise OSError("Hermes launch failed")
    with pytest.raises(OSError):
        supervise(engine, {**engine, "argv": ["hermes"]}, spawn=spawn, wait=lambda *_: None, stop=stopped.append)
    assert stopped == [server]


def test_scripted_engine_loss_stops_owned_hermes_then_engine():
    engine = {"argv": ["engine"], "cwd": "/unused", "env": {}, "endpoint": "http://127.0.0.1:9999/v1", "timeout": 10}
    server = SimpleNamespace(poll=lambda: 2)
    agent = SimpleNamespace(poll=lambda: None)
    stopped = []
    def spawn(argv, **_):
        return server if argv == ["engine"] else agent
    with pytest.raises(RuntimeError, match="Engine stopped"):
        supervise(engine, {**engine, "argv": ["hermes"]}, spawn=spawn, wait=lambda *_: None, stop=stopped.append)
    assert stopped == [agent, server]
