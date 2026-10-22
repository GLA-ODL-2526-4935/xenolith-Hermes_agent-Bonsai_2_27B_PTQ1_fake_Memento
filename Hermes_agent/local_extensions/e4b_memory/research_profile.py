"""Owner-configured research validation; independent of OS isolation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchValidator:
    enabled: bool = True

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("research_only.validator_enabled must be boolean")

    @classmethod
    def from_config(cls, config):
        settings = config.get("research_only", {})
        if not isinstance(settings, dict) or set(settings) - {"validator_enabled"}:
            raise ValueError("Invalid research-only settings")
        return cls(settings.get("validator_enabled", True))

    def validate(self, config, names=()):
        if not self.enabled:
            return
        allowed = {"e4b_memory", "e4b_browser"}
        if len(names) != len(set(names)) or set(names) - allowed:
            raise PermissionError("Research validator refuses non-research tools")
        toolsets = config.get("platform_toolsets", {})
        if not isinstance(toolsets, dict) or any(
                not isinstance(items, list) or set(items) - {"memory", "e4b_browser"}
                for items in toolsets.values()):
            raise PermissionError("Research validator refuses non-research toolsets")
        plugins = config.get("plugins", {}).get("enabled", [])
        if not isinstance(plugins, list) or set(plugins) - allowed or config.get("mcp_servers"):
            raise PermissionError("Research validator refuses other plugins/MCP servers")
