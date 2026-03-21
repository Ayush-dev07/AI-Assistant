from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from core.logging import get_logger
from memory.long_term import LongTermMemory, MemoryEntry, MemorySearchResult
from Security.injection_detector import InjectionDetector, ThreatLevel

if TYPE_CHECKING:
    from Security.audit_log import AuditLog, ActionType

log = get_logger(__name__)

_AGENT_SOURCES = frozenset({"agent", "self", "reflection", "synthesis"})

class GuardedMemory:

    def __init__(
        self,
        memory:    LongTermMemory,
        detector:  InjectionDetector,
        audit_log: "AuditLog | None" = None,
    ) -> None:
        self._memory    = memory
        self._detector  = detector
        self._audit     = audit_log

        self._stats: dict[str, int] = {
            "scan_safe":        0,
            "scan_suspicious":  0,
            "scan_blocked":     0,
            "store_calls":      0,
            "retrieve_calls":   0,
            "batch_calls":      0,
        }

    async def store(
        self,
        content:  str,
        source:   str          = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Scan content for injection, then store in LongTermMemory if safe.
        """
        self._stats["store_calls"] += 1
        
        scan_result = await self._detector.scan_detailed(content, source=source)
        threat      = scan_result.threat_level
        
        if threat == ThreatLevel.DANGEROUS:
            self._stats["scan_blocked"] += 1
            log.warning(
                "guarded_memory_store_blocked",
                source          = source,
                threat          = threat.value,
                content_preview = content[:100],
                matched_pattern = scan_result.matched_pattern[:60] if scan_result.matched_pattern else "",
            )
            await self._record_block(content, source, scan_result)
            return None

        if threat == ThreatLevel.SUSPICIOUS:
            self._stats["scan_suspicious"] += 1
            log.info(
                "guarded_memory_store_suspicious",
                source          = source,
                content_preview = content[:100],
            )
        else:
            self._stats["scan_safe"] += 1

        trust_level = _compute_trust_level(source, threat)
        
        augmented_meta: dict[str, Any] = dict(metadata or {})
        augmented_meta["source"]       = source
        augmented_meta["trust_level"]  = trust_level
        augmented_meta["threat_level"] = threat.value

        entry = MemoryEntry(
            content     = content,
            source      = source,
            trust_level = trust_level,
            metadata    = augmented_meta,
        )
        vector_id = await self._memory.store(entry)
        log.debug(
            "guarded_memory_stored",
            source      = source,
            trust_level = trust_level,
            threat      = threat.value,
            vector_id   = (vector_id[:8] + "...") if vector_id else None,
        )
        return vector_id

    async def store_batch(
        self,
        items: list[tuple[str, str, dict[str, Any] | None]],
    ) -> list[str | None]:
        self._stats["batch_calls"] += 1
        
        scan_tasks = [
            self._detector.scan_detailed(content, source=src)
            for content, src, _ in items
        ]
        scan_results = await asyncio.gather(*scan_tasks)
        
        store_tasks = []
        indices: list[int] = []

        for idx, ((content, source, metadata), scan_result) in enumerate(
            zip(items, scan_results)
        ):
            if scan_result.threat_level == ThreatLevel.DANGEROUS:
                self._stats["scan_blocked"] += 1
                await self._record_block(content, source, scan_result)
                store_tasks.append(asyncio.coroutine(lambda: None)())
                indices.append(-1)  
            else:
                store_tasks.append(
                    self.store(content=content, source=source, metadata=metadata)
                )
                indices.append(idx)

        raw_results = await asyncio.gather(*store_tasks)
        
        output: list[str | None] = [None] * len(items)
        for raw_idx, result in enumerate(raw_results):
            if indices[raw_idx] != -1:
                output[indices[raw_idx]] = result
        return output

    async def retrieve(
        self,
        query:          str,
        k:              int   = 5,
        min_trust:      float = 0.0,
        min_similarity: float = 0.3,
        source_filter:  str | None = None,
    ) -> list[MemorySearchResult]:
        self._stats["retrieve_calls"] += 1
        results = await self._memory.retrieve(
            query          = query,
            k              = k,
            min_trust      = min_trust,
            min_similarity = min_similarity,
            source_filter  = source_filter,
        )
        log.debug(
            "guarded_memory_retrieved",
            query_preview = query[:60],
            results       = len(results),
            min_trust     = min_trust,
        )
        return results

    async def retrieve_as_context(
        self,
        query:     str,
        k:         int   = 5,
        min_trust: float = 0.5,
    ) -> str:
        return await self._memory.retrieve_as_context(
            query     = query,
            k         = k,
            min_trust = min_trust,
        )

    async def _record_block(
        self,
        content:     str,
        source:      str,
        scan_result: Any,
    ) -> None:
        if self._audit is None:
            return
        try:
            from Security.audit_log import AuditEntry, ActionType
            await self._audit.record(AuditEntry(
                session_id     = "system",   
                action_type    = ActionType.MEMORY_WRITE,
                result_summary = (
                    f"BLOCKED: memory write rejected. "
                    f"source={source} "
                    f"threat=dangerous "
                    f"content_preview={content[:150]!r}"
                ),
                threat_level   = ThreatLevel.DANGEROUS.value,
            ))
        except Exception as exc:
            log.debug("guarded_memory_audit_failed", error=str(exc))

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0

    @property
    def memory(self) -> LongTermMemory:
        return self._memory

    def __repr__(self) -> str:
        return (
            f"<GuardedMemory "
            f"blocked={self._stats['scan_blocked']} "
            f"suspicious={self._stats['scan_suspicious']} "
            f"safe={self._stats['scan_safe']}>"
        )

def _compute_trust_level(source: str, threat: ThreatLevel) -> float:
    if source in _AGENT_SOURCES:
        return 1.0
    if threat == ThreatLevel.DANGEROUS:
        return 0.0   
    if threat == ThreatLevel.SUSPICIOUS:
        return 0.3
    return 0.8   