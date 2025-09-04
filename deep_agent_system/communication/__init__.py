"""Agent communication system for inter-agent messaging and coordination."""

from .manager import AgentCommunicationManager
from .coordinator import AgentCoordinator

__all__ = ["AgentCommunicationManager", "AgentCoordinator"]