"""Claude-backed agents that use tools and return validated pydantic models."""

from agents.base_agent import BaseAgent
from agents.investigator import InvestigatorAgent
from agents.proposer import ProposerAgent

__all__ = ["BaseAgent", "InvestigatorAgent", "ProposerAgent"]
