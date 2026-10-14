import json
import subprocess
import sys
from pathlib import Path

import pytest

from toolfit import check_tools, load_tools, make_strict
from toolfit.cli import main

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def codes(issues):
    return {i.code for i in issues}


def openai_tool(name="t", params=None, desc="does things"):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": params or {"type": "object", "properties": {}}}}


# ---------------------------------------------------------------- loader

def test_detects_all_shapes():
    schema = {"type": "object", "properties": {}}
    docs = {
        "openai-chat": [openai_tool()],
        "openai-responses": [{"type": "function", "name": "t", "parameters": schema}],
        "anthropic": [{"name": "t", "description": "d", "input_schema": schema}],
        "mcp": {"tools": [{"name": "t", "inputSchema": schema}]},
    }
    for shape, doc in docs.items():
        tools = load_tools(doc)
        assert len(tools) == 1 and tools[0].shape == shape and tools[0].name == "t"


def test_jsonrpc_result_and_request_body():
    rpc = {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "a", "inputSchema": {"type": "object"}}]}}
    assert load_tools(rpc)[0].shape == "mcp"
    body = {"model": "deepseek-chat", "messages": [], "tools": [openai_tool("x")]}
    assert load_tools(body)[0].name == "x"


def test_bare_schema():
    tools = load_tools({"type": "object", "properties": {"a": {"type": "string"}}})
    assert tools[0].shape == "json-schema"


def test_rejects_garbage():
    with pytest.raises(ValueError):
        load_tools([{"foo": 1}])
    with pytest.raises(ValueError):
        load_tools(42)


# ---------------------------------------------------------------- generic rules

def test_generic_rules():
    params = {
        "type": "object",
        "properties": {
            "a": {"type": "str"},
            "b": {"type": "array"},
            "c": {"type": "string", "enum": ["x"], "default": "y"},
            "d": {"$ref": "#/$defs/Missing"},
            "e": {"enum": []},
        },
        "required": ["a", "zzz"],
    }
    tools = load_tools([openai_tool("t", params, desc=""), openai_tool("t")])
    got = codes(check_tools(tools, ["generic"]))
    assert {"TF002", "TF003", "TF004", "TF005", "TF006", "TF007", "TF008", "TF009", "TF012"} <= got


def test_resolved_ref_ok():
    params = {"type": "object", "properties": {"f": {"$ref": "#/$defs/F"}}, "required": ["f"],
              "$defs": {"F": {"type": "object", "properties": {}, "description": "f"}}}
    assert "TF006" not in codes(check_tools(load_tools([openai_tool("t", params)]), ["generic"]))


def test_root_must_be_object():
    tools = load_tools([openai_tool("t", {"type": "string"})])
    assert "TF010" in codes(check_tools(tools, ["generic"]))


def test_token_budget():
    params = {"type": "object", "properties": {f"p{i}": {"type": "string", "description": "x" * 50} for i in range(40)}}
    tools = load_tools([openai_tool("big", params)])
    assert "TF011" in codes(check_tools(tools, ["generic"], token_budget=100))
    assert "TF011" not in codes(check_tools(tools, ["generic"], token_budget=100000))


# ---------------------------------------------------------------- provider rules

def test_name_rules():
    tools = load_tools([openai_tool("search.docs"), openai_tool("x" * 65), openai_tool("ok_name-1")])
    issues = [i for i in check_tools(tools, ["openai"]) if i.code == "TF020"]
    assert {i.tool for i in issues} == {"search.docs", "x" * 65}


def test_mcp_portability():
    tools = load_tools({"tools": [{"name": "gh.create", "inputSchema": {"type": "object"}},
                                  {"name": "bad name", "inputSchema": {"type": "object"}}]})
    got = check_tools(tools, ["mcp"])
    assert ("TF022", "gh.create") in {(i.code, i.tool) for i in got}
    assert ("TF021", "bad name") in {(i.code, i.tool) for i in got}


def test_tool_count_limit():
    tools = load_tools([openai_tool(f"t{i}") for i in range(129)])
    assert "TF030" in codes(check_tools(tools, ["openai"]))
    assert "TF030" not in codes(check_tools(tools[:128], ["openai"]))


def test_strict_rules():
    params = {"type": "object", "properties": {
        "a": {"type": "string", "minLength": 2},
        "b": {"type": "object", "properties": {"c": {"type": "integer"}}, "allOf": []},
    }, "required": ["a"]}
    got = codes(check_tools(load_tools([openai_tool("t", params)]), ["openai-strict"]))
    assert {"TF041", "TF042", "TF044", "TF045"} <= got


def test_strict_depth():
    inner = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    for _ in range(11):
        inner = {"type": "object", "properties": {"n": inner}, "required": ["n"],
                 "additionalProperties": False, "description": "n"}
    got = codes(check_tools(load_tools([openai_tool("t", inner)]), ["openai-strict"]))
    assert "TF043" in got


def test_unknown_profile():
    with pytest.raises(ValueError):
        check_tools([], ["nope"])


# ---------------------------------------------------------------- fix

def test_make_strict_passes_strict_profile():
    doc = json.loads((EX / "openai_tools.json").read_text())
    for t in load_tools(doc):
        t.holder[t.schema_key] = make_strict(t.schema)
    strict_codes = {"TF040", "TF041", "TF042", "TF043", "TF044", "TF046"}
    assert not (codes(check_tools(load_tools(doc), ["openai-strict"])) & strict_codes)


def test_make_strict_nullable_shapes():
    s = make_strict({"type": "object", "properties": {
        "a": {"type": "string"}, "b": {"type": ["integer"]}, "c": {"$ref": "#/$defs/C"},
        "d": {"anyOf": [{"type": "string"}]}, "e": {"type": "string"}},
        "required": ["e", "ghost"], "$defs": {"C": {"type": "object", "properties": {"x": {"type": "string"}}}}})
    p = s["properties"]
    assert p["a"]["type"] == ["string", "null"]
    assert p["b"]["type"] == ["integer", "null"]
    assert p["c"] == {"anyOf": [{"$ref": "#/$defs/C"}, {"type": "null"}]}
    assert {"type": "null"} in p["d"]["anyOf"]
    assert p["e"]["type"] == "string"  # was already required: untouched
    assert s["required"] == ["e", "a", "b", "c", "d"]
    assert s["additionalProperties"] is False
    assert s["$defs"]["C"]["additionalProperties"] is False


def test_make_strict_does_not_mutate_input():
    original = {"type": "object", "properties": {"a": {"type": "string"}}}
    make_strict(original)
    assert original == {"type": "object", "properties": {"a": {"type": "string"}}}


# ---------------------------------------------------------------- CLI

def test_cli_exit_codes(capsys):
    assert main(["check", str(EX / "openai_tools.json"), "-p", "openai"]) == 1
    assert main(["check", str(EX / "mcp_tools_list.json"), "-p", "mcp"]) == 0
    assert main(["check", str(EX / "mcp_tools_list.json"), "-p", "mcp", "--fail-on", "warning"]) == 1
    assert main(["check", str(EX / "openai_tools.json"), "--fail-on", "never"]) == 0


def test_cli_json_output(capsys):
    main(["check", str(EX / "openai_tools.json"), "-p", "openai-strict", "-f", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["tools"] == 2 and data["estimated_tokens"] > 0
    assert any(i["code"] == "TF041" for i in data["issues"])


def test_cli_github_output(capsys):
    main(["check", str(EX / "openai_tools.json"), "-p", "openai", "-f", "github"])
    out = capsys.readouterr().out
    assert "::error file=" in out and "TF020" in out


def test_cli_fix_roundtrip(tmp_path):
    out = tmp_path / "fixed.json"
    assert main(["fix", str(EX / "openai_tools.json"), "-o", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert all(t["function"]["strict"] is True for t in doc)
    assert main(["check", str(out), "-p", "openai-strict", "--fail-on", "never"]) == 0


def test_cli_bad_input(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert main(["check", str(bad)]) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_module_entrypoint_stdin():
    doc = json.dumps([openai_tool("fine")])
    r = subprocess.run([sys.executable, "-m", "toolfit", "check", "-", "-p", "deepseek"],
                       input=doc, capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert r.returncode == 0, r.stderr
    assert "1 tool(s)" in r.stdout
