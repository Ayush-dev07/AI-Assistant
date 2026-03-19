from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from core.llm.base import LLMMessage
from core.logging import get_logger

if TYPE_CHECKING:
    from core.llm.base import LLMProvider
    from agents.base import AgentResult

log = get_logger(__name__)

_NUM_PATTERN = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?")

_MIN_SIGNIFICANT_NUM = 10.0

def _extract_group1(m: re.Match) -> str:
    return m.group(1).strip()

_ENTITY_PATTERNS: list[tuple[str, re.Pattern, Any]] = [
    (
        "date",
        re.compile(
            r"\b(\d{4}-\d{1,2}-\d{1,2}"       
            r"|\w+ \d{1,2},? \d{4}"            
            r"|\d{1,2}/\d{1,2}/\d{2,4})\b",    
            re.IGNORECASE,
        ),
        _extract_group1,
    ),
    (
        "attribution",
        re.compile(
            r"(?:ceo|founder|founded by|cto|president|director|created by)"
            r"\s+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            re.IGNORECASE,
        ),
        _extract_group1,
    ),
    (
        "source",
        re.compile(
            r"(?:according to|reported by|source:)\s+"
            r"([A-Z][A-Za-z0-9&'\s]{2,40}?)(?:\s*[,\.;]|\s+(?:the|a|an|that))",
            re.IGNORECASE,
        ),
        _extract_group1,
    ),
    (
        "currency_figure",
        re.compile(
            r"[\$€£¥](\d+(?:[.,]\d+)?(?:\s*[KMBTkmbt](?:illion|illion)?)?\b)",
        ),
        _extract_group1,
    ),
]

class ResultAggregator:

    def __init__(
        self,
        llm:                         "LLMProvider",
        synthesis_temperature:       float = 0.5,
        conflict_confidence_penalty: float = 0.15,
        max_context_chars_per_agent: int   = 4000,
    ) -> None:
        self._llm               = llm
        self._synth_temp        = synthesis_temperature
        self._conflict_penalty  = conflict_confidence_penalty
        self._max_context_chars = max_context_chars_per_agent

    async def aggregate_results(
        self,
        goal:    str,
        results: dict[str, "AgentResult"],
    ) -> tuple[str, float]:
        if not results:
            return "No sub-agent results to synthesise.", 0.0

        if len(results) == 1:
            only_id     = next(iter(results))
            only_result = results[only_id]
            text        = (
                only_result.to_context_string()
                if hasattr(only_result, "to_context_string")
                else str(getattr(only_result, "output", ""))
            )
            confidence = getattr(only_result, "confidence", 1.0)
            return text, confidence

        texts:       dict[str, str]   = {}
        confidences: dict[str, float] = {}

        for node_id, result in results.items():
            if hasattr(result, "to_context_string"):
                text = result.to_context_string()
            else:
                text = str(getattr(result, "output", ""))

            texts[node_id]       = text[: self._max_context_chars]
            confidences[node_id] = float(getattr(result, "confidence", 1.0))

        return await self._run_synthesis(goal, texts, confidences)

    async def aggregate(
        self,
        goal:    str,
        results: dict[str, str],
    ) -> str:
        if not results:
            return "No results to synthesise."
        if len(results) == 1:
            return next(iter(results.values()))

        texts = {k: v[: self._max_context_chars] for k, v in results.items()}
        answer, _ = await self._run_synthesis(goal, texts, {})
        return answer

    def detect_conflicts(
        self,
        results: dict[str, str],
    ) -> list[str]:
        numeric  = self._detect_numeric_conflicts(results)
        entities = self._detect_entity_conflicts(results)
        return numeric + entities

    def _detect_numeric_conflicts(
        self,
        results: dict[str, str],
    ) -> list[str]:
        number_sets: dict[str, set[float]] = {}
        for node_id, text in results.items():
            raw_nums = _NUM_PATTERN.findall(str(text))
            normalised: set[float] = set()
            for n in raw_nums:
                try:
                    val = float(n.replace(",", ""))
                    if val >= _MIN_SIGNIFICANT_NUM:
                        normalised.add(val)
                except ValueError:
                    pass
            number_sets[node_id] = normalised

        conflicts: list[str] = []
        ids = list(number_sets.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b    = ids[i], ids[j]
                diff_a  = number_sets[a] - number_sets[b]
                diff_b  = number_sets[b] - number_sets[a]

                if diff_a or diff_b:
                    a_str = sorted(diff_a) if diff_a else []
                    b_str = sorted(diff_b) if diff_b else []
                    conflicts.append(
                        f"{a} has {a_str} that {b} does not; "
                        f"{b} has {b_str} that {a} does not  [numbers]"
                    )

        return conflicts

    def _detect_entity_conflicts(
        self,
        results: dict[str, str],
    ) -> list[str]:
        conflicts: list[str] = []
        ids = list(results.keys())

        for entity_type, pattern, extract_fn in _ENTITY_PATTERNS:
            entity_sets: dict[str, set[str]] = {}
            for node_id, text in results.items():
                entities = set()
                for m in pattern.finditer(str(text)):
                    try:
                        val = extract_fn(m).lower().strip()
                        if val:
                            entities.add(val)
                    except Exception:
                        pass
                entity_sets[node_id] = entities

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b   = ids[i], ids[j]
                    diff_a = entity_sets.get(a, set()) - entity_sets.get(b, set())
                    diff_b = entity_sets.get(b, set()) - entity_sets.get(a, set())

                    if diff_a or diff_b:
                        conflicts.append(
                            f"{a} mentions {entity_type}s {sorted(diff_a)} "
                            f"that {b} does not; "
                            f"{b} mentions {sorted(diff_b)} that {a} does not "
                            f"[{entity_type}]"
                        )

        return conflicts

    async def _run_synthesis(
        self,
        goal:        str,
        texts:       dict[str, str],
        confidences: dict[str, float],
    ) -> tuple[str, float]:
        """
        Call the LLM to synthesise texts into a hedged final answer.
        """
        conflicts     = self.detect_conflicts(texts)
        conflict_text = _format_conflicts(conflicts)
        results_block = _format_results(texts, confidences)

        has_conflicts = len(conflicts) > 0
        hedge_instruction = (
            "\nIMPORTANT: Conflicts were detected (see above). "
            "You MUST acknowledge these discrepancies explicitly. "
            "Do NOT silently choose one value over another. "
            "State both values and indicate which source provided each."
            if has_conflicts
            else "\nNo significant conflicts detected. Synthesise confidently."
        )

        prompt = (
            f"Synthesise the following sub-agent results for the goal below.\n\n"
            f"GOAL: {goal}\n\n"
            f"--- SUB-AGENT RESULTS ---\n"
            f"{results_block}\n\n"
            f"--- CONFLICT ANALYSIS ---\n"
            f"{conflict_text}\n"
            f"{hedge_instruction}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Produce a complete, well-structured final answer.\n"
            f"- Cite which agent provided each key fact.\n"
            f"- If a result has low confidence, note the uncertainty.\n"
            f"- Start your response with: FINAL ANSWER:\n"
        )

        log.debug(
            "aggregator_synthesis_start",
            goal_len      = len(goal),
            agent_count   = len(texts),
            conflict_count = len(conflicts),
            llm_model     = self._llm.model_name,
        )

        try:
            response = await self._llm.complete(
                messages    = [LLMMessage(role="user", content=prompt)],
                temperature = self._synth_temp,
                max_tokens  = 4096,
            )
            text = response.content.strip()
            if text.upper().startswith("FINAL ANSWER:"):
                text = text[len("FINAL ANSWER:"):].strip()

        except Exception as exc:
            log.warning(
                "aggregator_synthesis_failed",
                error = str(exc),
            )
            text = "\n\n".join(
                f"## {node_id}\n{content}"
                for node_id, content in texts.items()
            )

        if confidences:
            avg_conf = sum(confidences.values()) / len(confidences)
        else:
            avg_conf = 1.0

        conflict_pairs = sum(
            1 for c in conflicts
            if " vs " in c or " has " in c
        ) // 2  

        confidence = max(
            0.0,
            avg_conf - (conflict_pairs * self._conflict_penalty)
        )

        log.debug(
            "aggregator_synthesis_complete",
            answer_len      = len(text),
            conflicts       = len(conflicts),
            avg_confidence  = round(avg_conf, 3),
            final_confidence = round(confidence, 3),
        )

        return text, round(confidence, 3)

def _format_results(
    texts:       dict[str, str],
    confidences: dict[str, float],
) -> str:
    parts = []
    for node_id, text in texts.items():
        conf = confidences.get(node_id)
        conf_str = f"  [confidence: {conf:.2f}]" if conf is not None else ""
        parts.append(f"[{node_id}]{conf_str}:\n{text}")
    return "\n\n".join(parts)


def _format_conflicts(conflicts: list[str]) -> str:
    if not conflicts:
        return "No significant conflicts detected across agent results."
    lines = ["CONFLICTS DETECTED — must acknowledge in final answer:"]
    for c in conflicts:
        lines.append(f"  ⚠  {c}")
    return "\n".join(lines)