"""Opt-in section-first local discovery; never broadens the caller's tool grant."""
from __future__ import annotations

import copy
import json

from tools.registry import registry, tool_error


def groups(tool_defs):
    sections = {}
    for definition in tool_defs:
        function = definition.get("function") or {}
        name = function.get("name")
        entry = registry.get_entry(name) if isinstance(name, str) else None
        if entry is not None:
            sections.setdefault(entry.toolset, []).append(function)
    return {key: sorted(value, key=lambda fn: fn["name"]) for key, value in sorted(sections.items())}


def schemas(tool_defs):
    sections = list(groups(tool_defs))
    hint = "Available sections: " + ", ".join(sections) + "." if sections else "No sections available."
    descriptions = {
        "tool_search": "Discover tool sections or catalog ONE section: names and brief descriptions only. " + hint,
        "tool_describe": "Get the full format of ONE chosen tool within its section. No other schemas are returned.",
        "tool_call": "Invoke one tool using its described format. Existing session permissions, approvals and hooks still apply.",
    }
    properties = {
        "tool_search": {"action": {"type": "string", "enum": ["sections", "catalog"]}, "section": {"type": "string"}},
        "tool_describe": {"section": {"type": "string"}, "name": {"type": "string"}},
        "tool_call": {"calls": {"type": "array", "minItems": 1, "maxItems": 1, "items": {
            "type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["name", "arguments"], "additionalProperties": False}}},
    }
    required = {"tool_search": ["action"], "tool_describe": ["section", "name"], "tool_call": ["calls"]}
    return [{"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties[name], "required": required[name], "additionalProperties": False}}}
            for name, description in descriptions.items()]


def search(args, tool_defs):
    if not isinstance(args, dict) or set(args) - {"action", "section"}:
        return tool_error("Use action=sections or action=catalog with one exact section")
    sections = groups(tool_defs)
    if args.get("action") == "sections":
        return json.dumps({"sections": [{"name": key, "count": len(value)} for key, value in sections.items()]})
    section = args.get("section")
    if args.get("action") != "catalog" or not isinstance(section, str) or section not in sections:
        return tool_error("Unknown section or action in this session")
    tools = [{"name": fn["name"], "description": " ".join(str(fn.get("description", "")).split())[:160]}
             for fn in sections[section]]
    if len(tools) > 100 or len(json.dumps(tools)) > 24000:
        return tool_error("Section exceeds catalog budget; owner must split this toolset before use")
    return json.dumps({"section": section, "tools": tools}, ensure_ascii=False)


def describe(args, tool_defs):
    if not isinstance(args, dict) or set(args) != {"section", "name"}:
        return tool_error("Select one section and one exact tool name")
    if not isinstance(args["section"], str) or not isinstance(args["name"], str):
        return tool_error("Section and name must be strings")
    sections = groups(tool_defs)
    function = next((fn for fn in sections.get(args["section"], []) if fn["name"] == args["name"]), None)
    if function is None:
        return tool_error("Tool is not available in this session/section")
    if len(json.dumps(function)) > 48000:
        return tool_error("Tool schema exceeds budget; owner must narrow it")
    return json.dumps({"tool": copy.deepcopy(function)}, ensure_ascii=False)
