"""Owned-process lifetime and loopback readiness for the explicit target launcher."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def child_environment(profile):
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "XDG_RUNTIME_DIR",
               "XDG_DATA_DIRS", "DISPLAY", "WAYLAND_DISPLAY", "LD_LIBRARY_PATH", "VK_DRIVER_FILES", "VK_ICD_FILENAMES"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(HERMES_HOME=profile, HERMES_RUNTIME_DIR=str(profile) + "/runtime")
    return env


def free_endpoint(endpoint):
    url = urlsplit(endpoint)
    if url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port:
        raise ValueError("Only explicit loopback engine endpoints are allowed")
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex((url.hostname, url.port)) == 0:
            raise PermissionError("Endpoint occupied; never attach to or stop an unrelated server")


def wait_ready(process, endpoint, timeout):
    from .sorting import NoRedirect
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with opener.open(endpoint + "/models", timeout=2) as response:
                raw = response.read(16001)
                if len(raw) > 16000:
                    raise ValueError("Oversized model identity response")
                body = json.loads(raw)
                if any(row.get("id") == "bonsai-2-27b" for row in body.get("data", []) if isinstance(row, dict)):
                    return
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            pass
        threading.Event().wait(0.5)
    raise RuntimeError("Engine exited or readiness timed out; Hermes was not started")


def stop_owned(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def supervise(engine, agent, *, wait=wait_ready, spawn=subprocess.Popen, stop=stop_owned):
    """Test seam: all subprocesses are owned; never retry model/tool requests."""
    children = []
    prior = {}

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            prior[sig] = signal.signal(sig, interrupted)
        server = spawn(engine["argv"], cwd=engine["cwd"], env=engine["env"],
                       stdin=subprocess.DEVNULL, start_new_session=True)
        children.append(server)
        wait(server, engine["endpoint"], engine["timeout"])
        hermes = spawn(agent["argv"], cwd=agent["cwd"], env=agent["env"], start_new_session=True)
        children.append(hermes)
        while hermes.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("Engine stopped; stopping owned Hermes session")
            threading.Event().wait(0.5)
        return hermes.returncode
    finally:
        for child in reversed(children):
            stop(child)
        for sig, handler in prior.items():
            signal.signal(sig, handler)


def run_stack(plan, repo):
    from .store import file_lock
    from .launch_stack import prepare_profile, validate_runtime
    root = repo / "Hermes_agent"
    profile = Path(plan["profile"])
    with file_lock(profile.parent / (".stack-" + profile.name + ".lock")):
        free_endpoint(plan["endpoint"])
        validate_runtime(plan)
        prepare_profile(plan)
        environment = child_environment(plan["profile"])
        engine = {"argv": plan["engine_argv"], "cwd": str(repo / "engine"), "env": environment,
                  "endpoint": plan["endpoint"], "timeout": plan["timeout"]}
        agent = {"argv": plan["hermes_argv"], "cwd": str(root), "env": environment}
        return supervise(engine, agent)
