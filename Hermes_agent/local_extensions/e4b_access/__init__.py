"""Approval broker foundations; no auto-activation, browser or system tools."""
import json

from .client import Client
from .surface import BRIDGE, dispatch


def register(ctx):
    def client():
        from agent.secret_scope import get_secret
        if ctx.get_config("enabled", False) is not True:
            raise PermissionError("Access bridge is disabled")
        return Client(ctx.get_config("endpoint"), get_secret("E4B_ACCESS_CLIENT_TOKEN"))

    def handle(args, **_kwargs):
        try:
            if ctx.get_config("enabled", False) is not True:
                raise PermissionError("Access bridge is disabled")
            return json.dumps({"success": True, **dispatch(args, client, scope=ctx.get_config("scope", "manual"))}, ensure_ascii=False)
        except (ValueError, PermissionError, TypeError, OSError):
            return json.dumps({"success": False, "error": "Access request refused"})

    ctx.register_tool(name="e4b_access", toolset="e4b_access", schema=BRIDGE, handler=handle,
                      check_fn=lambda: ctx.get_config("enabled", False) is True)
