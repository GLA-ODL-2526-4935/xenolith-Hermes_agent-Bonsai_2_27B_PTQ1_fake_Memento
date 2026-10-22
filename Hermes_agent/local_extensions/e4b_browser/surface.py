"""Progressive browser discovery: no full operation formats in the initial bridge."""
import copy

from local_extensions.e4b_memory.research_profile import ResearchValidator


def schema(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


STRING = {"type": "string"}
BRIDGE = {"name": "e4b_browser", "description": (
    "Research browser. Discover sections, catalog operations, describe one, then call it. "
    "Open returns metadata only; outline and read fetch bounded sections on demand."),
    "parameters": schema({"action": {"type": "string", "enum": ["sections", "catalog", "describe", "call"]},
                          "section": STRING, "name": STRING, "arguments": {"type": "object"}}, ["action"])}
SECTIONS = {"navigation": ("search", "open"), "reading": ("outline", "read")}
DESCRIPTIONS = {"search": "Search the single connected browser; return at most five result titles/URLs.",
                "open": "Open a public HTTPS page; return metadata, not the whole page.",
                "outline": "List a few section headings; optionally match a query against section text.",
                "read": "Read a bounded verbatim section chunk; continue with its opaque handle."}
SCHEMAS = {
    "search": schema({"query": STRING, "limit": {"type": "integer", "minimum": 1, "maximum": 5}}, ["query"]),
    "open": schema({"url": STRING}, ["url"]),
    "outline": schema({"page_id": STRING, "query": STRING, "cursor": STRING,
                       "limit": {"type": "integer", "minimum": 1, "maximum": 8}}, ["page_id"]),
    "read": schema({"page_id": STRING, "section_id": STRING, "cursor": STRING,
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 3000}}, ["page_id", "section_id"])}


class BrowserSurface:
    def __init__(self, session, config, *, tool_names=("e4b_memory", "e4b_browser")):
        self.session = session
        self.config = copy.deepcopy(config)
        self.validator = ResearchValidator.from_config(self.config)
        self.tool_names = tuple(tool_names)
        self.validator.validate(self.config, self.tool_names)
        self.described = set()

    def dispatch(self, args):
        self.validator.validate(self.config, self.tool_names)
        if not isinstance(args, dict) or set(args) - {"action", "section", "name", "arguments"}:
            raise ValueError("Invalid browser arguments")
        action = args.get("action")
        if action == "sections":
            return self.session.deliver({"sections": list(SECTIONS)})
        names = SECTIONS.get(args.get("section"))
        if names is None:
            raise ValueError("Unknown browser section")
        if action == "catalog":
            return self.session.deliver({"operations": [{"name": name, "description": DESCRIPTIONS[name]} for name in names]})
        name = args.get("name")
        if name not in names:
            raise PermissionError("Browser operation is unavailable")
        if action == "describe":
            result = self.session.deliver({"name": name, "parameters": copy.deepcopy(SCHEMAS[name])})
            self.described.add(name)
            return result
        if action != "call" or name not in self.described:
            raise PermissionError("Describe the browser operation before calling it")
        values = args.get("arguments")
        spec = SCHEMAS[name]
        if not isinstance(values, dict) or set(values) - set(spec["properties"]) or set(spec["required"]) - set(values):
            raise ValueError("Invalid browser operation arguments")
        # Fixed vocabulary, not arbitrary getattr/evaluate or model-selected backend.
        operations = {"search": self.session.search, "open": self.session.open,
                      "outline": self.session.outline, "read": self.session.read}
        return operations[name](**values)
