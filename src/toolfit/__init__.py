"""toolfit: lint LLM tool / function-calling schemas before the API rejects them."""

__version__ = "0.1.0"

from .loader import Tool, load_tools  # noqa: E402
from .rules import Issue, PROFILES, check_tools  # noqa: E402
from .fix import make_strict  # noqa: E402

__all__ = ["Tool", "load_tools", "Issue", "PROFILES", "check_tools", "make_strict", "__version__"]
