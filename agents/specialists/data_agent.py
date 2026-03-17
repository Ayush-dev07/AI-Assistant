from __future__ import annotations

from typing import Any

from agents.base import BaseAgent, try_parse_json, extract_confidence

DATA_SYSTEM_PROMPT = """\
You are a Data Agent. Your tools are: filesystem (for parsing data files)
and code_executor (for analysis).

Your workflow for every task:
1. PARSE: use filesystem with action=parse to read the data file.
   This gives you headers, sample rows, and basic statistics for free.
2. ANALYSE: write Python code using the parsed data to compute what
   was asked. Use only: json, csv, math, statistics, collections,
   datetime, re, itertools, functools.
3. EXECUTE: run the analysis code with code_executor.
4. INTERPRET: read the stdout carefully and extract key findings.
5. DETECT ANOMALIES: flag any values more than 3 std deviations from
   the mean, missing data > 5% of column, or unexpected value ranges.
6. CHART DESCRIPTIONS: describe 1-2 charts that would best visualise
   the findings (you do not need to render them — just describe them).

CRITICAL RULES:
- Always parse the file first before writing analysis code.
- Use only the column names exactly as they appear in the parsed headers.
- Do not hard-code values — compute everything from the data.
- Report exact numbers, not vague descriptions ("revenue increased" is
  not useful — "revenue increased 23.4%, from $1.2M to $1.48M" is).

You MUST end with: FINAL ANSWER:
Then output JSON:
{
  "summary": "2-3 sentence overview of the key findings",
  "key_metrics": {
    "metric_name": value,
    ...
  },
  "anomalies": ["description of each anomaly, or empty list"],
  "chart_descriptions": [
    "Chart 1: bar chart of X vs Y showing ...",
    "Chart 2: line chart of ..."
  ],
  "schema": {
    "columns": ["col1", "col2", ...],
    "row_count": 0,
    "file_format": "csv/xlsx"
  },
  "confidence": 0.0-1.0
}
"""


class DataAgent(BaseAgent):

    agent_type    = "data"
    capabilities  = [
        "csv_parsing",
        "xlsx_parsing",
        "statistical_analysis",
        "anomaly_detection",
        "chart_description",
        "schema_inference",
    ]
    default_tools  = ["filesystem", "code_executor"]
    system_prompt  = DATA_SYSTEM_PROMPT
    max_iterations = 6

    def _parse_output(self, raw: str) -> tuple[Any, float]:
        parsed, ok = try_parse_json(raw)
        if ok and parsed:
            confidence = extract_confidence(parsed)

            key_metrics = parsed.get("key_metrics", {})
            anomalies   = parsed.get("anomalies", [])
            schema      = parsed.get("schema", {})

            if not key_metrics:
                confidence *= 0.8   
            if schema.get("row_count", 0) == 0:
                confidence *= 0.9  

            output = {
                "summary":            parsed.get("summary", ""),
                "key_metrics":        key_metrics,
                "anomalies":          anomalies,
                "chart_descriptions": parsed.get("chart_descriptions", []),
                "schema":             schema,
                "confidence":         round(confidence, 3),
            }
            return output, confidence

        import re
        numbers = re.findall(r"(\w[\w\s]*?):\s*([\d.,]+%?)", raw)
        metrics = {k.strip(): v.strip() for k, v in numbers[:10]}

        return {
            "summary":            raw[:1000] if raw else "No analysis produced.",
            "key_metrics":        metrics,
            "anomalies":          [],
            "chart_descriptions": [],
            "schema":             {},
            "confidence":         0.45,
        }, 0.45