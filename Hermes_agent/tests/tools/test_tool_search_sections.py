"""Section-first discovery through the real tool-search assembly/dispatcher."""
import json
from unittest.mock import patch

import pytest

from tools.registry import registry
from tools import tool_search as ts


def register(name, section, *, handler=None, description="Brief description", properties=None):
    definition = {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {"chosen_format_marker": {"type": "string"}},
        "required": ["chosen_format_marker"], "additionalProperties": False}}}
    registry.register(name=name, toolset=section, schema=definition,
                      handler=handler or (lambda args, **kw: json.dumps({"success": True, "echo": args})))
    return definition


@pytest.fixture
def definitions():
    return [register("section_test_read", "section-test-reports"),
            register("section_test_other", "section-test-other", properties={"other_format_marker": {"type": "string"}})]


@pytest.fixture
def config():
    return ts.ToolSearchConfig.from_raw({"enabled": "on", "discovery_mode": "sections"})


def test_no_upfront_tool_names_descriptions_or_parameter_formats(definitions, config):
    result = ts.assemble_tool_defs(definitions, config=config)
    encoded = json.dumps(result.tool_defs)
    assert result.activated and result.listing_form == "sections"
    assert "section-test-reports" in encoded
    for marker in ("section_test_read", "section_test_other", "chosen_format_marker", "other_format_marker", "Brief description"):
        assert marker not in encoded
    assert {d["function"]["name"] for d in result.tool_defs} == ts.BRIDGE_TOOL_NAMES


def test_catalog_and_describe_return_only_selected_section_and_tool(definitions, config):
    result = json.loads(ts.dispatch_tool_search({"action": "catalog", "section": "section-test-reports"},
                                               current_tool_defs=definitions, config=config))
    assert result["tools"] == [{"name": "section_test_read", "description": "Brief description"}]
    assert "chosen_format_marker" not in json.dumps(result)
    described = json.loads(ts.dispatch_tool_describe({"section": "section-test-reports", "name": "section_test_read"},
                                                    current_tool_defs=definitions, config=config))
    assert "chosen_format_marker" in json.dumps(described)
    assert "other_format_marker" not in json.dumps(described)


def test_catalog_is_session_scoped_and_no_global_tool_leak(definitions, config):
    result = json.loads(ts.dispatch_tool_search({"action": "sections"}, current_tool_defs=definitions[:1], config=config))
    assert result["sections"] == [{"name": "section-test-reports", "count": 1}]
    for args in ({"section": "section-test-other", "name": "section_test_other"},
                 {"section": "section-test-reports", "name": "section_test_other"}):
        assert "error" in json.loads(ts.dispatch_tool_describe(args, current_tool_defs=definitions[:1], config=config))


def test_discovery_does_not_mutate_the_schema_prefix(definitions, config):
    before = json.dumps(ts.assemble_tool_defs(definitions, config=config).tool_defs)
    ts.dispatch_tool_search({"action": "catalog", "section": "section-test-reports"}, current_tool_defs=definitions, config=config)
    ts.dispatch_tool_describe({"section": "section-test-reports", "name": "section_test_read"}, current_tool_defs=definitions, config=config)
    assert json.dumps(ts.assemble_tool_defs(definitions, config=config).tool_defs) == before


def test_invalid_calls_do_not_expose_other_sections(definitions, config):
    for args in ({"action": "catalog", "section": []}, {"queries": ["everything"]}, {"action": "export"}):
        assert "error" in json.loads(ts.dispatch_tool_search(args, current_tool_defs=definitions, config=config))
    for args in ({"names": ["section_test_read", "section_test_other"]}, {"section": [], "name": "section_test_read"}):
        assert "error" in json.loads(ts.dispatch_tool_describe(args, current_tool_defs=definitions, config=config))


def test_large_catalog_and_schema_fail_with_actionable_budget_error(config):
    definition = register("section_test_large", "section-test-big", description="x" * 49000)
    assert "error" in json.loads(ts.dispatch_tool_describe({"section": "section-test-big", "name": "section_test_large"},
                                                          current_tool_defs=[definition], config=config))
    many = [register(f"section_test_many_{n}", "section-test-many") for n in range(101)]
    assert "error" in json.loads(ts.dispatch_tool_search({"action": "catalog", "section": "section-test-many"},
                                                       current_tool_defs=many, config=config))


def test_default_and_off_modes_are_unchanged(definitions):
    assert ts.ToolSearchConfig.from_raw(None).discovery_mode == "search"
    assert ts.ToolSearchConfig.from_raw({"discovery_mode": "typo"}).discovery_mode == "search"
    config = ts.ToolSearchConfig.from_raw({"enabled": "off", "discovery_mode": "sections"})
    assert ts.assemble_tool_defs(definitions, config=config).tool_defs == definitions


def test_config_yaml_loader_drives_section_assembly(definitions):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"tools": {"tool_search": {
        "enabled": "on", "discovery_mode": "sections"}}}), encoding="utf-8")
    assert ts.load_config_readonly().discovery_mode == "sections"
    assert ts.assemble_tool_defs(definitions).listing_form == "sections"


def test_core_tool_deferral_keeps_inline_tools_direct(config):
    register("section_test_core", "section-test-core")
    with patch("tools.tool_search._core_tool_names", return_value=frozenset({"section_test_core", "memory"})):
        assert ts.is_deferrable_tool_name("section_test_core", config.effective_defer_tools)
        assert not ts.is_deferrable_tool_name("memory", config.effective_defer_tools)


def test_section_call_uses_real_dispatch_and_preserves_approval_hooks(config):
    import model_tools
    from hermes_cli.plugins import get_plugin_manager
    observed = []
    definition = register("section_test_execution", "section-test-execution",
                          handler=lambda args, **kw: json.dumps({"success": True, "echo": args["chosen_format_marker"]}))
    manager = get_plugin_manager()

    def deny(tool_name, **kwargs):
        observed.append(tool_name)
        return {"action": "block", "message": "Owner refused"}

    with patch("tools.tool_search.load_config_readonly", return_value=config), patch("tools.tool_search.load_config", return_value=config):
        result = json.loads(model_tools.handle_function_call("tool_call", {"calls": [{
            "name": "section_test_execution", "arguments": {"chosen_format_marker": "ok"}}]},
            enabled_toolsets=["section-test-execution"]))
        assert result["echo"] == "ok"
        with patch.dict(manager._hooks, {"pre_tool_call": [deny]}):
            blocked = model_tools.handle_function_call("tool_call", {"calls": [{
                "name": "section_test_execution", "arguments": {"chosen_format_marker": "blocked"}}]},
                enabled_toolsets=["section-test-execution"])
        assert "Owner refused" in blocked and "section_test_execution" in observed
        assert definition["function"]["name"] in ts.scoped_deferrable_names([definition])


def test_section_call_refuses_out_of_session_and_connector_names(config):
    import model_tools
    register("section_test_denied", "section-test-denied")
    with patch("tools.tool_search.load_config_readonly", return_value=config), patch("tools.tool_search.load_config", return_value=config):
        result = model_tools.handle_function_call("tool_call", {"calls": [{"name": "section_test_denied",
            "arguments": {"chosen_format_marker": "no"}}]}, enabled_toolsets=["section-test-different"])
        assert "not available" in result.lower()
        _name, _args, error = ts.resolve_underlying_call({"calls": [{"name": "connectors__private__read", "arguments": {}}]})
        assert error and "hosted connectors" in error
