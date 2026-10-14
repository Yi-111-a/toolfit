"""Rewrite schemas so they pass OpenAI strict mode."""

from __future__ import annotations

import copy
from typing import Any, Dict

from .rules import is_object_schema


def _nullable(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    t = schema.get("type")
    if isinstance(t, str):
        if t != "null":
            schema["type"] = [t, "null"]
            if "enum" in schema and None not in schema["enum"]:
                schema["enum"] = list(schema["enum"]) + [None]
        return schema
    if isinstance(t, list):
        if "null" not in t:
            schema["type"] = list(t) + ["null"]
        return schema
    if isinstance(schema.get("anyOf"), list):
        if not any(isinstance(s, dict) and s.get("type") == "null" for s in schema["anyOf"]):
            schema["anyOf"].append({"type": "null"})
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def _walk(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    props = schema.get("properties")
    if isinstance(props, dict):
        for k in list(props):
            props[k] = _walk(props[k])
    if isinstance(schema.get("items"), dict):
        schema["items"] = _walk(schema["items"])
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            schema[key] = [_walk(s) for s in schema[key]]
    for key in ("$defs", "definitions"):
        if isinstance(schema.get(key), dict):
            for k in list(schema[key]):
                schema[key][k] = _walk(schema[key][k])
    if is_object_schema(schema):
        schema.setdefault("type", "object")
        schema["additionalProperties"] = False
        props = schema.setdefault("properties", {})
        required = [r for r in (schema.get("required") or []) if r in props]
        for k in props:
            if k not in required:
                required.append(k)
                props[k] = _nullable(props[k])
        schema["required"] = required
    return schema


def make_strict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *schema* rewritten for OpenAI strict mode.

    * every object gets ``"additionalProperties": false``
    * every property becomes required; previously optional ones become nullable
      (``["type", "null"]``) so the model can still omit a value by sending ``null``.
    * ``required`` entries that name non-existent properties are dropped.
    """
    return _walk(copy.deepcopy(schema))
