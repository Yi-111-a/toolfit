"""Detect and extract tool definitions from the common JSON shapes.

Supported shapes (auto-detected, can be mixed in a list):

* OpenAI Chat Completions: ``{"type": "function", "function": {"name", "description", "parameters"}}``
* OpenAI Responses API (flat): ``{"type": "function", "name", "description", "parameters"}``
* Anthropic Messages: ``{"name", "description", "input_schema"}``
* MCP ``tools/list`` result: ``{"tools": [{"name", "description", "inputSchema"}]}``
* A request body containing ``"tools": [...]`` (OpenAI / Anthropic / DeepSeek)
* A bare JSON Schema (treated as one anonymous tool's parameters)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SCHEMA_KEYS = ("parameters", "input_schema", "inputSchema")


@dataclass
class Tool:
    name: Optional[str]
    description: Optional[str]
    schema: Any
    holder: Dict[str, Any]  # dict that owns the schema key (mutated by fixes)
    schema_key: str
    shape: str
    source: str
    index: int

    @property
    def label(self) -> str:
        return self.name if self.name else f"<tool #{self.index}>"


def _from_entry(entry: Any, source: str, index: int) -> Optional[Tool]:
    if not isinstance(entry, dict):
        return None
    fn = entry.get("function")
    if entry.get("type") == "function" and isinstance(fn, dict):
        if "parameters" not in fn:
            fn["parameters"] = {"type": "object", "properties": {}}
        return Tool(fn.get("name"), fn.get("description"), fn["parameters"], fn,
                    "parameters", "openai-chat", source, index)
    for key, shape in (("parameters", "openai-responses"), ("input_schema", "anthropic"),
                       ("inputSchema", "mcp")):
        if key in entry and ("name" in entry or "description" in entry):
            return Tool(entry.get("name"), entry.get("description"), entry[key], entry,
                        key, shape, source, index)
    if "name" in entry and not any(k in entry for k in SCHEMA_KEYS) and "type" not in entry:
        # tool with no parameters at all
        entry["parameters"] = {"type": "object", "properties": {}}
        return Tool(entry.get("name"), entry.get("description"), entry["parameters"], entry,
                    "parameters", "openai-responses", source, index)
    return None


def load_tools(doc: Any, source: str = "<input>") -> List[Tool]:
    """Return every tool definition found in a parsed JSON document."""
    if isinstance(doc, dict) and isinstance(doc.get("tools"), list):
        entries = doc["tools"]
    elif isinstance(doc, dict) and isinstance(doc.get("result"), dict) and \
            isinstance(doc["result"].get("tools"), list):  # raw JSON-RPC response
        entries = doc["result"]["tools"]
    elif isinstance(doc, list):
        entries = doc
    elif isinstance(doc, dict):
        tool = _from_entry(doc, source, 0)
        if tool:
            return [tool]
        if "properties" in doc or doc.get("type") == "object" or "$schema" in doc:
            wrapper = {"schema": doc}
            return [Tool(doc.get("title"), doc.get("description"), doc, wrapper, "schema",
                         "json-schema", source, 0)]
        raise ValueError(f"{source}: could not find any tool definitions")
    else:
        raise ValueError(f"{source}: expected a JSON object or array")

    tools: List[Tool] = []
    for i, entry in enumerate(entries):
        tool = _from_entry(entry, source, i)
        if tool is None:
            raise ValueError(f"{source}: entry #{i} does not look like a tool definition")
        tools.append(tool)
    return tools
