"""Context-gathering tools for cloning repositories and searching their source.

These helpers are repo-agnostic: callers pass the working directory, target
slug, file extensions, and search query. Nothing here assumes a particular
language, layout, or framework.
"""

from tools.code_search import CodeSearchTool

__all__ = ["CodeSearchTool"]
