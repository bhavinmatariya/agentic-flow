"""Claude-backed agents that use tools and return validated pydantic models."""

from agents.base_agent import BaseAgent
from agents.investigator import InvestigatorAgent

__all__ = ["BaseAgent", "InvestigatorAgent"]
