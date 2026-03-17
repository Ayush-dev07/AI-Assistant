from agents.base import AgentResult, BaseAgent
from agents.orchestrator import Orchestrator, OrchestratorResult, TaskNode
from agents.specialists import CodingAgent, CommsAgent, DataAgent, ResearchAgent
from core.llm.base import LLMProvider

def build_default_agents(llm: "LLMProvider") -> dict[str, BaseAgent]: 
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from core.llm.base import LLMProvider

    return {
        "research":      ResearchAgent(llm=llm),
        "coding":        CodingAgent(llm=llm),
        "communication": CommsAgent(llm=llm),
        "data":          DataAgent(llm=llm),
    }


__all__ = [
    # Core abstractions
    "BaseAgent",
    "AgentResult",
    # Orchestrator
    "Orchestrator",
    "OrchestratorResult",
    "TaskNode",
    # Specialists
    "ResearchAgent",
    "CodingAgent",
    "CommsAgent",
    "DataAgent",
    # Factory
    "build_default_agents",
]