"""Explicit single-user test launcher; never invoked by plugin import."""
import argparse
import ipaddress
import json
from pathlib import Path
import shutil
import sys
import uuid

from .sorting import local_endpoint
from .store import file_lock, text


def network_policy(event, args):
    """Fail-closed Python network guard. Not a substitute for an OS sandbox."""
    if event == "socket.getaddrinfo":
        host = args[0]
        if host == "localhost":
            return
    elif event == "socket.connect":
        address = args[1]
        if not isinstance(address, tuple):
            raise PermissionError("Non-IP sockets are disabled in the memory test launcher")
        host = address[0]
    else:
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except (ValueError, TypeError):
        pass
    raise PermissionError("Only loopback networking is permitted in the memory test launcher")


def profile_config(base_url, model, *, semantic_sort=False, memory_service_url=None, shadow_capture=False,
                   research_validator=True):
    from .research_profile import ResearchValidator
    ResearchValidator(research_validator)
    if not isinstance(semantic_sort, bool):
        raise ValueError("semantic_sort must be boolean")
    if type(shadow_capture) is not bool:
        raise ValueError("shadow_capture must be boolean")
    endpoint = local_endpoint(base_url)
    text(model, 200)
    base = endpoint.removesuffix("/chat/completions")
    route = {"provider": "custom", "model": model, "base_url": base, "fallback_chain": []}
    config = {"model": {"provider": "custom", "default": model, "base_url": base},
            "memory": {"provider": "e4b_memory", "memory_enabled": False, "user_profile_enabled": False,
                       "e4b_memory": {"semantic_sort": semantic_sort, "l0_days": 30, "recovery_days": 7}},
            "platform_toolsets": {"cli": ["memory"]}, "fallback_models": [],
            "research_only": {"validator_enabled": research_validator},
            "mcp_servers": {}, "plugins": {"enabled": ["e4b_memory"]},
            "auxiliary": {key: dict(route) for key in (
                "compression", "vision", "title", "session_search", "curator", "review", "memory")}}
    if shadow_capture:
        config["memory"]["e4b_memory"]["shadow_capture"] = True
    if memory_service_url is not None:
        from .remote import validate_endpoint
        if semantic_sort:
            raise ValueError("Configure semantic sorting on the service owner side, not the agent profile")
        config["memory"]["e4b_memory"].update(backend="service", service_url=validate_endpoint(memory_service_url))
    return config


def create_profile(destination, base_url, model, *, semantic_sort=False, memory_service_url=None, shadow_capture=False,
                   research_validator=True):
    config = profile_config(base_url, model, semantic_sort=semantic_sort, memory_service_url=memory_service_url,
                            shadow_capture=shadow_capture, research_validator=research_validator)
    destination = Path(destination).resolve()
    source = Path(__file__).parent.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Profile must be outside the plugin source directory")
    destination.mkdir(parents=True, exist_ok=False)
    # JSON is a strict YAML subset; no new configuration parser/dependency is needed.
    (destination / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    shutil.copytree(source, destination / "plugins" / "e4b_memory",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination


def validate_surface(agent):
    names = [tool.get("function", {}).get("name") for tool in agent.tools]
    if names != ["e4b_memory"]:
        raise PermissionError(f"Refusing non-memory tool surface: {names}")
    manager = getattr(agent, "_memory_manager", None)
    if not manager or [p.name for p in manager.providers] != ["e4b_memory"]:
        raise PermissionError("Required memory provider is not active")


def run_profile(profile):
    # A dedicated, unmodified generated profile is required; this never edits a live profile.
    profile = Path(profile).resolve()
    actual = json.loads((profile / "config.yaml").read_text(encoding="utf-8"))
    from .research_profile import ResearchValidator
    validator = ResearchValidator.from_config(actual)
    model = actual["model"]
    memory = actual.get("memory", {}).get("e4b_memory", {})
    expected = profile_config(model["base_url"], model["default"], semantic_sort=memory.get("semantic_sort", False),
                              memory_service_url=memory.get("service_url"), shadow_capture=memory.get("shadow_capture", False),
                              research_validator=validator.enabled)
    if "research_only" not in actual:  # Existing generated memory-only profiles keep working.
        expected.pop("research_only")
    if actual != expected:
        raise PermissionError("Restricted launcher requires its generated, unmodified profile")
    validator.validate(actual)
    from .endpoint_policy import profile_endpoints
    sys.addaudithook(profile_endpoints(actual))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(profile)
    agent = None
    try:
        with file_lock(profile / "single-owner.lock"):
            from run_agent import AIAgent
            agent = AIAgent(model=model["default"], base_url=model["base_url"], provider="custom",
                            api_key="local-only", api_mode="chat_completions", platform="cli",
                            session_id=uuid.uuid4().hex, enabled_toolsets=["memory"],
                            skip_context_files=True, skip_background_review=True, save_trajectories=False)
            validate_surface(agent)
            validator.validate(actual, [tool["function"]["name"] for tool in agent.tools])
            print(f"Memory-only session: {agent.session_id}. Type /quit to exit.")
            while True:
                message = input("> ").strip()
                if message == "/quit":
                    break
                if message:
                    validate_surface(agent)
                    validator.validate(actual, [tool["function"]["name"] for tool in agent.tools])
                    print(agent.chat(message))
    finally:
        if agent is not None:
            agent.close()
        reset_hermes_home_override(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    init = commands.add_parser("init")
    init.add_argument("path")
    init.add_argument("--base-url", required=True)
    init.add_argument("--model", required=True)
    init.add_argument("--semantic-sort", action="store_true", help="Use the same local model for idle layer classification")
    init.add_argument("--shadow-capture", action="store_true", help="Record bounded visible-text checkpoints; no context replacement")
    init.add_argument("--memory-service-url", help="Opt into remote storage; credentials are scoped secrets, not CLI arguments")
    init.add_argument("--no-research-validator", action="store_true",
                      help="Disable only the extra profile validator; memory-only launcher restrictions remain")
    run = commands.add_parser("run")
    run.add_argument("path")
    args = parser.parse_args()
    if args.action == "init":
        print(create_profile(args.path, args.base_url, args.model, semantic_sort=args.semantic_sort,
                              memory_service_url=args.memory_service_url, shadow_capture=args.shadow_capture,
                              research_validator=not args.no_research_validator))
    else:
        run_profile(args.path)


if __name__ == "__main__":
    main()
