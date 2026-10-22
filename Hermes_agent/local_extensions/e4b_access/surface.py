"""One staged proposal bridge; no owner decisions or arbitrary file paths."""
import copy

from .policy import MAX_TEXT, WEB_ACTIONS, validate_proposal


def schema(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


STRING = {"type": "string"}
BRIDGE = {"name": "e4b_access", "description": (
    "Request owner-controlled browser handoffs or text exports. Discover sections, catalog, then describe "
    "one operation for its format. Requests are NOT executed browser actions. No system access or self-approval."),
    "parameters": schema({"action": {"type": "string", "enum": ["sections", "catalog", "describe", "call"]},
                          "section": STRING, "name": STRING, "arguments": {"type": "object"}}, ["action"])}
SECTIONS = {"browser": WEB_ACTIONS, "files": ("text_export",), "requests": ("status",)}
DESCRIPTIONS = {**{name: "Ask owner to review this exact browser action; manual handoff only." for name in WEB_ACTIONS},
                "text_export": "Ask owner to create one plain-text export; no model-selected path or overwrite.",
                "status": "Check one request ID; no inbox listing, content read or approval."}


def operation_schema(name):
    if name == "status":
        return schema({"id": STRING}, ["id"])
    if name == "text_export":
        return schema({"label": {"type": "string", "maxLength": 120},
                       "text": {"type": "string", "maxLength": MAX_TEXT}}, ["label", "text"])
    return schema({"url": {"type": "string", "maxLength": 4096},
                   "details": {"type": "string", "maxLength": 4000}}, ["url", "details"])


def dispatch(args, client_factory, *, scope="manual"):
    if scope not in {"manual", "research"}:
        raise PermissionError("Unknown access scope")
    sections = SECTIONS if scope == "manual" else {"browser": ("open_page",), "requests": ("status",)}
    if not isinstance(args, dict) or set(args) - {"action", "section", "name", "arguments"}:
        raise ValueError("Invalid bridge arguments")
    action = args.get("action")
    if action == "sections":
        return {"sections": list(sections)}
    names = sections.get(args.get("section"))
    if names is None:
        raise ValueError("Unknown section")
    if action == "catalog":
        return {"operations": [{"name": name, "description": DESCRIPTIONS[name]} for name in names]}
    name = args.get("name")
    if name not in names:
        raise PermissionError("Unknown operation")
    if action == "describe":
        return {"name": name, "parameters": copy.deepcopy(operation_schema(name))}
    if action != "call" or not isinstance(args.get("arguments"), dict):
        raise ValueError("Invalid operation call")
    values = args["arguments"]
    if name == "status":
        if set(values) != {"id"}:
            raise ValueError("Status requires only an ID")
        body = {"action": "status", "id": values["id"]}
    else:
        if "kind" in values:
            raise ValueError("Operation cannot be overridden")
        body = {"action": "submit", "proposal": validate_proposal({"kind": name, **values})}
    return client_factory().call(body)
