"""Claude-backed agents that use tools and return validated pydantic models.

Import agents from their modules directly (for example
``from agents.investigator import InvestigatorAgent``). This package
``__init__`` does not eagerly import submodules to avoid loading the full
agent graph when only one agent is needed.
"""

__all__ = [
    "BaseAgent",
    "ImplementerAgent",
    "InvestigatorAgent",
    "ProposerAgent",
    "ResponseParserAgent",
    "ReviewerAgent",
]
