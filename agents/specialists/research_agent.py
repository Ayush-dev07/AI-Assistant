from __future__ import annotations

from typing import Any

from agents.base import BaseAgent, try_parse_json, extract_confidence

RESEARCH_SYSTEM_PROMPT = """\
You are a Research Agent. Your only tool is the browser.

Your job for every task:
1. Use the browser to search for information (action='search', query=...).
2. Fetch the 2-3 most relevant URLs from the search results (action='fetch').
3. Cross-reference facts across sources — note any contradictions.
4. Produce a structured JSON response.

CRITICAL RULES:
- Never state a fact without citing the URL it came from.
- If two sources contradict each other, report BOTH values and flag it.
- If you cannot find reliable information, say so — do not guess.
- Prefer primary sources (official docs, papers, company blogs) over aggregators.

You MUST end your final answer with this exact JSON structure:
{
  "summary": "2-3 sentence overview",
  "key_facts": ["fact with [source: url]", ...],
  "sources": ["https://url1", "https://url2", ...],
  "confidence": 0.0-1.0,
  "contradictions": ["description if any, else empty list"]
}

Start your final answer with: FINAL ANSWER:
Then output the JSON.
"""


class ResearchAgent(BaseAgent):
    agent_type    = "research"
    capabilities  = [
        "web_search",
        "page_fetch",
        "information_synthesis",
        "fact_checking",
        "source_citation",
        "contradiction_detection",
    ]
    default_tools = ["browser"]
    system_prompt = RESEARCH_SYSTEM_PROMPT
    max_iterations = 5

    def _parse_output(self, raw: str) -> tuple[Any, float]:
        parsed, ok = try_parse_json(raw)
        if ok and parsed:
            confidence = extract_confidence(parsed)
            output = {
                "summary":        parsed.get("summary", raw[:500]),
                "key_facts":      parsed.get("key_facts", []),
                "sources":        parsed.get("sources", []),
                "confidence":     confidence,
                "contradictions": parsed.get("contradictions", []),
            }
            return output, confidence

        return {
            "summary":        raw[:2000] if raw else "No output produced.",
            "key_facts":      [],
            "sources":        [],
            "confidence":     0.6,
            "contradictions": [],
        }, 0.6