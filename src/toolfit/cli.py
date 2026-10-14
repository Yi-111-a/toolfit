"""Command-line interface for toolfit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional, Sequence, Tuple

from . import __version__
from .fix import make_strict
from .loader import Tool, load_tools
from .rules import PROFILES, SEVERITY_RANK, Issue, check_tools, estimate_tokens

COLORS = {"error": "\033[31m", "warning": "\033[33m", "info": "\033[36m"}
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"


def _read(path: str) -> Tuple[Any, str]:
    if path == "-":
        text, name = sys.stdin.read(), "<stdin>"
    else:
        with open(path, encoding="utf-8") as fh:
            text, name = fh.read(), path
    try:
        return json.loads(text), name
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}: invalid JSON ({exc})") from None


def _use_color(stream: Any) -> bool:
    return hasattr(stream, "isatty") and stream.isatty() and not os.environ.get("NO_COLOR")


def _format_text(issues: List[Issue], tools: List[Tool], show_info: bool, color: bool) -> str:
    lines = []
    for i in issues:
        if i.severity == "info" and not show_info:
            continue
        sev = i.severity
        if color:
            sev = f"{COLORS[sev]}{sev}{RESET}"
        where = f"{i.tool}{(' ' + i.path) if i.path else ''}"
        loc = f"{DIM}{i.source}{RESET}" if color else i.source
        lines.append(f"{loc}: {sev} {i.code} [{where}] {i.message}")
    counts = {s: sum(1 for i in issues if i.severity == s) for s in ("error", "warning", "info")}
    tokens = sum(estimate_tokens(t) for t in tools)
    summary = (f"{len(tools)} tool(s), ~{tokens} tokens of definitions: "
               f"{counts['error']} error(s), {counts['warning']} warning(s), "
               f"{counts['info']} info")
    if not show_info and counts["info"]:
        summary += " (use --info to show)"
    lines.append((BOLD + summary + RESET) if color else summary)
    return "\n".join(lines)


def _format_github(issues: List[Issue]) -> str:
    level = {"error": "error", "warning": "warning", "info": "notice"}
    out = []
    for i in issues:
        msg = f"{i.code} [{i.tool}{(' ' + i.path) if i.path else ''}] {i.message}"
        out.append(f"::{level[i.severity]} file={i.source},title=toolfit {i.code}::{msg}")
    return "\n".join(out)


def cmd_check(args: argparse.Namespace) -> int:
    profiles = args.profile or ["generic"]
    all_issues: List[Issue] = []
    all_tools: List[Tool] = []
    for path in args.files:
        doc, name = _read(path)
        tools = load_tools(doc, name)
        all_tools.extend(tools)
        issues = check_tools(tools, profiles, token_budget=args.token_budget)
        order = {t.label: t.index for t in tools}
        issues.sort(key=lambda i: order.get(i.tool, -1))
        all_issues.extend(issues)
    if args.format == "json":
        print(json.dumps({
            "profiles": profiles,
            "tools": len(all_tools),
            "estimated_tokens": sum(estimate_tokens(t) for t in all_tools),
            "issues": [i.to_dict() for i in all_issues],
        }, indent=2, ensure_ascii=False))
    elif args.format == "github":
        text = _format_github(all_issues)
        if text:
            print(text)
    else:
        print(_format_text(all_issues, all_tools, args.info, _use_color(sys.stdout)))
    if args.fail_on == "never":
        return 0
    threshold = SEVERITY_RANK[args.fail_on]
    return 1 if any(SEVERITY_RANK[i.severity] >= threshold for i in all_issues) else 0


def cmd_fix(args: argparse.Namespace) -> int:
    doc, name = _read(args.file)
    tools = load_tools(doc, name)
    for t in tools:
        if isinstance(t.schema, dict):
            t.holder[t.schema_key] = make_strict(t.schema)
            if t.shape in ("openai-chat", "openai-responses") and not args.no_strict_flag:
                t.holder["strict"] = True
    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    target = args.file if args.in_place else args.output
    if target and target != "-":
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {target} ({len(tools)} tool(s) rewritten for strict mode)", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    width = max(len(p) for p in PROFILES)
    for name, (desc, _) in PROFILES.items():
        print(f"{name.ljust(width)}  {desc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="toolfit",
        description="Lint LLM tool / function-calling schemas (OpenAI, Anthropic, DeepSeek, "
                    "MCP) before the API rejects them.")
    p.add_argument("--version", action="version", version=f"toolfit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="lint tool definitions")
    c.add_argument("files", nargs="+", help="JSON files ('-' for stdin)")
    c.add_argument("-p", "--profile", action="append", choices=list(PROFILES),
                   help="provider profile; repeatable (default: generic)")
    c.add_argument("-f", "--format", choices=["text", "json", "github"], default="text")
    c.add_argument("--fail-on", choices=["error", "warning", "info", "never"], default="error",
                   help="exit 1 when an issue at or above this severity exists (default: error)")
    c.add_argument("--token-budget", type=int, default=600,
                   help="flag tools whose definition exceeds ~N tokens (default: 600)")
    c.add_argument("--info", action="store_true", help="also print info-level findings")
    c.set_defaults(func=cmd_check)

    f = sub.add_parser("fix", help="rewrite schemas to pass OpenAI strict mode")
    f.add_argument("file", help="JSON file ('-' for stdin)")
    out = f.add_mutually_exclusive_group()
    out.add_argument("-o", "--output", help="write result here (default: stdout)")
    out.add_argument("-i", "--in-place", action="store_true", help="overwrite the input file")
    f.add_argument("--no-strict-flag", action="store_true",
                   help="do not add \"strict\": true to OpenAI-shaped tools")
    f.set_defaults(func=cmd_fix)

    pr = sub.add_parser("profiles", help="list available profiles")
    pr.set_defaults(func=cmd_profiles)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        print(f"toolfit: {exc}", file=sys.stderr)
        return 2
