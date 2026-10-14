"""Lint rules for tool schemas, grouped into provider profiles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterator, List, Tuple

from .loader import Tool

ERROR, WARNING, INFO = "error", "warning", "info"
SEVERITY_RANK = {INFO: 0, WARNING: 1, ERROR: 2}

JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}

# OpenAI, DeepSeek (OpenAI-compatible) and Anthropic all document this pattern.
STRICT_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
# MCP spec (2025-11-25) guidance: 1-128 chars of A-Z a-z 0-9 _ - .
MCP_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

OPENAI_MAX_TOOLS = 128
STRICT_MAX_DEPTH = 10
STRICT_MAX_PROPERTIES = 5000
STRICT_UNSUPPORTED = (
    "allOf", "not", "if", "then", "else", "dependentRequired", "dependentSchemas",
    "patternProperties", "unevaluatedProperties", "propertyNames",
    "minProperties", "maxProperties",
)
STRICT_MAYBE_UNSUPPORTED = ("minLength", "maxLength")


@dataclass
class Issue:
    code: str
    severity: str
    tool: str
    path: str
    message: str
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- helpers

def iter_schemas(schema: Any, path: str = "#", depth: int = 1) -> Iterator[Tuple[Any, str, int]]:
    """Yield (subschema, json_pointer, object_depth) for every nested schema."""
    if not isinstance(schema, dict):
        return
    yield schema, path, depth
    is_obj = schema.get("type") == "object" or "properties" in schema
    child_depth = depth + 1 if is_obj else depth
    props = schema.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            yield from iter_schemas(v, f"{path}/properties/{k}", child_depth)
    items = schema.get("items")
    if isinstance(items, dict):
        yield from iter_schemas(items, f"{path}/items", depth)
    elif isinstance(items, list):
        for i, v in enumerate(items):
            yield from iter_schemas(v, f"{path}/items/{i}", depth)
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            for i, v in enumerate(schema[key]):
                yield from iter_schemas(v, f"{path}/{key}/{i}", depth)
    for key in ("not", "if", "then", "else"):
        if isinstance(schema.get(key), dict):
            yield from iter_schemas(schema[key], f"{path}/{key}", depth)
    ap = schema.get("additionalProperties")
    if isinstance(ap, dict):
        yield from iter_schemas(ap, f"{path}/additionalProperties", child_depth)
    for key in ("$defs", "definitions"):
        if isinstance(schema.get(key), dict):
            for k, v in schema[key].items():
                yield from iter_schemas(v, f"{path}/{key}/{k}", depth)


def is_object_schema(s: Dict[str, Any]) -> bool:
    t = s.get("type")
    return t == "object" or (isinstance(t, list) and "object" in t) or \
        ("properties" in s and "type" not in s)


def resolve_ref(root: Any, ref: str) -> bool:
    if not ref.startswith("#"):
        return True  # remote refs are out of scope
    node = root
    for part in [p for p in ref[1:].split("/") if p]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False
    return True


def estimate_tokens(tool: Tool) -> int:
    """Rough token estimate (~4 characters per token) of what the model sees."""
    payload = {"name": tool.name, "description": tool.description, "parameters": tool.schema}
    return max(1, len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False)) // 4)


# --------------------------------------------------------------------------- rules

Rule = Callable[[List[Tool], Dict[str, Any]], Iterator[Issue]]


def rule_generic(tools: List[Tool], opts: Dict[str, Any]) -> Iterator[Issue]:
    seen: Dict[str, int] = {}
    for t in tools:
        if not t.name:
            yield Issue("TF001", ERROR, t.label, "", "tool has no name", t.source)
        elif t.name in seen:
            yield Issue("TF002", ERROR, t.label, "",
                        f"duplicate tool name (also tool #{seen[t.name]})", t.source)
        else:
            seen[t.name] = t.index
        if not (t.description or "").strip() and t.shape != "json-schema":
            yield Issue("TF003", WARNING, t.label, "",
                        "tool has no description; models pick tools mostly by description",
                        t.source)
        root = t.schema
        if not isinstance(root, dict):
            yield Issue("TF010", ERROR, t.label, "#", "parameters must be a JSON object schema",
                        t.source)
            continue
        if root.get("type") != "object":
            yield Issue("TF010", ERROR, t.label, "#",
                        f"root parameters schema must have \"type\": \"object\" "
                        f"(got {json.dumps(root.get('type'))})", t.source)
        for s, path, _ in iter_schemas(root):
            typ = s.get("type")
            types = typ if isinstance(typ, list) else [typ] if typ is not None else []
            for ty in types:
                if ty not in JSON_TYPES:
                    yield Issue("TF007", ERROR, t.label, path,
                                f"invalid JSON Schema type {json.dumps(ty)}", t.source)
            props = s.get("properties")
            req = s.get("required")
            if isinstance(req, list):
                names = props if isinstance(props, dict) else {}
                for r in req:
                    if r not in names:
                        yield Issue("TF005", ERROR, t.label, path,
                                    f"\"required\" lists {json.dumps(r)} which is not in "
                                    f"\"properties\"", t.source)
            elif req is not None:
                yield Issue("TF005", ERROR, t.label, path, "\"required\" must be an array",
                            t.source)
            if isinstance(props, dict):
                for k, v in props.items():
                    if isinstance(v, dict) and "$ref" not in v and not v.get("description"):
                        yield Issue("TF004", INFO, t.label, f"{path}/properties/{k}",
                                    "parameter has no description", t.source)
            if "$ref" in s and isinstance(s["$ref"], str) and not resolve_ref(root, s["$ref"]):
                yield Issue("TF006", ERROR, t.label, path,
                            f"unresolved $ref {json.dumps(s['$ref'])}", t.source)
            if "enum" in s:
                enum = s["enum"]
                if not isinstance(enum, list) or not enum:
                    yield Issue("TF008", ERROR, t.label, path, "\"enum\" must be a non-empty array",
                                t.source)
                elif "default" in s and s["default"] not in enum:
                    yield Issue("TF012", WARNING, t.label, path,
                                f"default {json.dumps(s['default'])} is not one of the enum values",
                                t.source)
            if (typ == "array" or (isinstance(typ, list) and "array" in typ)) and "items" not in s:
                yield Issue("TF009", WARNING, t.label, path,
                            "array has no \"items\"; several providers reject or mis-handle this",
                            t.source)
        budget = opts.get("token_budget", 600)
        tokens = estimate_tokens(t)
        if tokens > budget:
            yield Issue("TF011", INFO, t.label, "",
                        f"definition is ~{tokens} tokens (budget {budget}); it is sent on every "
                        f"request", t.source)


def _name_rule(pattern: re.Pattern, desc: str, code: str, severity: str) -> Rule:
    def rule(tools: List[Tool], opts: Dict[str, Any]) -> Iterator[Issue]:
        for t in tools:
            if t.name and not pattern.match(t.name):
                yield Issue(code, severity, t.label, "",
                            f"name {json.dumps(t.name)} must match {desc}", t.source)
    return rule


rule_name_openai = _name_rule(STRICT_NAME_RE, "^[a-zA-Z0-9_-]{1,64}$", "TF020", ERROR)
rule_name_mcp = _name_rule(MCP_NAME_RE, "^[A-Za-z0-9_.-]{1,128}$ (MCP spec guidance)",
                           "TF021", WARNING)


def rule_mcp_portable(tools: List[Tool], opts: Dict[str, Any]) -> Iterator[Issue]:
    for t in tools:
        if t.name and MCP_NAME_RE.match(t.name) and not STRICT_NAME_RE.match(t.name):
            yield Issue("TF022", WARNING, t.label, "",
                        f"name {json.dumps(t.name)} is valid MCP but will be rejected by "
                        f"OpenAI/Anthropic/DeepSeek clients (dots or >64 chars)", t.source)


def rule_tool_count(tools: List[Tool], opts: Dict[str, Any]) -> Iterator[Issue]:
    if len(tools) > OPENAI_MAX_TOOLS:
        yield Issue("TF030", ERROR, "*", "",
                    f"{len(tools)} tools; OpenAI accepts at most {OPENAI_MAX_TOOLS} per request",
                    tools[0].source)


def rule_openai_strict(tools: List[Tool], opts: Dict[str, Any]) -> Iterator[Issue]:
    for t in tools:
        root = t.schema
        if not isinstance(root, dict):
            continue
        if "anyOf" in root:
            yield Issue("TF040", ERROR, t.label, "#",
                        "strict mode: root schema must be an object, not anyOf", t.source)
        total_props = 0
        for s, path, depth in iter_schemas(root):
            if is_object_schema(s):
                if s.get("additionalProperties") is not False:
                    yield Issue("TF041", ERROR, t.label, path,
                                "strict mode: object needs \"additionalProperties\": false",
                                t.source)
                props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
                total_props += len(props)
                req = s.get("required") if isinstance(s.get("required"), list) else []
                missing = [k for k in props if k not in req]
                if missing:
                    yield Issue("TF042", ERROR, t.label, path,
                                "strict mode: every property must be required (use a "
                                "[\"type\", \"null\"] union for optional ones); not required: "
                                + ", ".join(missing), t.source)
                if depth > STRICT_MAX_DEPTH:
                    yield Issue("TF043", ERROR, t.label, path,
                                f"strict mode: nesting depth {depth} exceeds {STRICT_MAX_DEPTH}",
                                t.source)
            for kw in STRICT_UNSUPPORTED:
                if kw in s:
                    yield Issue("TF044", ERROR, t.label, path,
                                f"strict mode: keyword \"{kw}\" is not supported", t.source)
            for kw in STRICT_MAYBE_UNSUPPORTED:
                if kw in s:
                    yield Issue("TF045", WARNING, t.label, path,
                                f"strict mode: \"{kw}\" is not in OpenAI's supported keyword "
                                f"list; it may be rejected or ignored", t.source)
        if total_props > STRICT_MAX_PROPERTIES:
            yield Issue("TF046", ERROR, t.label, "#",
                        f"strict mode: {total_props} object properties exceeds "
                        f"{STRICT_MAX_PROPERTIES}", t.source)


PROFILES: Dict[str, Tuple[str, List[Rule]]] = {
    "generic": ("Provider-neutral hygiene checks", [rule_generic]),
    "openai": ("OpenAI Chat Completions / Responses tools",
               [rule_generic, rule_name_openai, rule_tool_count]),
    "openai-strict": ("OpenAI with \"strict\": true (Structured Outputs)",
                      [rule_generic, rule_name_openai, rule_tool_count, rule_openai_strict]),
    "anthropic": ("Anthropic Messages API tools", [rule_generic, rule_name_openai]),
    "deepseek": ("DeepSeek (OpenAI-compatible) tools",
                 [rule_generic, rule_name_openai, rule_tool_count]),
    "mcp": ("MCP server tools (tools/list), incl. portability to LLM APIs",
            [rule_generic, rule_name_mcp, rule_mcp_portable]),
}


def check_tools(tools: List[Tool], profiles: List[str], **opts: Any) -> List[Issue]:
    rules: List[Rule] = []
    for p in profiles:
        if p not in PROFILES:
            raise ValueError(f"unknown profile {p!r}; choose from {', '.join(PROFILES)}")
        for r in PROFILES[p][1]:
            if r not in rules:
                rules.append(r)
    issues: List[Issue] = []
    seen = set()
    for r in rules:
        for issue in r(tools, opts):
            key = (issue.code, issue.tool, issue.path, issue.message)
            if key not in seen:
                seen.add(key)
                issues.append(issue)
    return issues
