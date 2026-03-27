from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm.base import LLMProvider
    from memory.long_term import LongTermMemory


def build_security_stack(
    llm:         "LLMProvider | None" = None,
    long_memory: "LongTermMemory | None" = None,
) -> dict[str, Any]:
    from Security.injection_detector import InjectionDetector
    from Security.audit_log          import AuditLog
    from Security.anomaly_detector   import AnomalyDetector
    from memory.guarded_memory       import GuardedMemory

    if llm is None:
        from core.llm.gemini import GeminiProvider
        llm = GeminiProvider()

    detector = InjectionDetector(llm=llm)
    audit    = AuditLog()       
    anomaly  = AnomalyDetector()  

    guarded: GuardedMemory | None = None
    if long_memory is not None:
        guarded = GuardedMemory(
            memory    = long_memory,
            detector  = detector,
            audit_log = audit,
        )
    return {
        "detector": detector,
        "audit":    audit,
        "anomaly":  anomaly,
        "guarded":  guarded,
    }

from Security.injection_detector import InjectionDetector, ThreatLevel
from Security.audit_log          import AuditLog, AuditEntry, ActionType
from Security.anomaly_detector   import AnomalyDetector, CircuitState, Thresholds
from Security.hitl               import (
    ApprovalGate, requires_approval, build_request,
    sanitize_parameters, ACTIONS_REQUIRING_APPROVAL,
)
__all__ = [
    "build_security_stack",
    
    "InjectionDetector",
    "ThreatLevel",
    
    "AuditLog",
    "AuditEntry",
    "ActionType",
    
    "AnomalyDetector",
    "CircuitState",
    "Thresholds",
    
    "ApprovalGate",
    "requires_approval",
    "build_request",
    "sanitize_parameters",
    "ACTIONS_REQUIRING_APPROVAL",
]