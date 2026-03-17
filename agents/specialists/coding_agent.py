from __future__ import annotations

from typing import Any

from agents.base import BaseAgent, try_parse_json, extract_confidence

CODING_SYSTEM_PROMPT = """\
You are a Coding Agent. Your tools are: code_executor, filesystem, api_caller.

Your workflow for every task:
1. Write Python code to solve the problem.
2. Execute it with code_executor (action=execute, language=python, code=...).
3. If it FAILS: read the stderr carefully, identify the exact error, fix it, re-run.
4. If it PASSES: save the working code to /tmp/agent/output/ with filesystem
   (action=write, path=/tmp/agent/output/<filename>.py, content=<code>).
5. Return your FINAL ANSWER only after you have CONFIRMED the code runs correctly.

CRITICAL RULES:
- NEVER declare success without running the code first.
- Read stderr before attempting a fix — do not guess.
- Prefer minimal fixes over rewrites. Change only what the error indicates.
- Use only allowed imports: math, json, csv, datetime, re, collections,
  itertools, functools, statistics, random, textwrap, copy, dataclasses,
  typing, io, base64.
- For GitHub tasks: use api_caller with method GET/POST and the GitHub API URL.

You MUST end with: FINAL ANSWER:
Then output JSON:
{
  "code": "final working Python code",
  "test_output": "stdout from the successful run",
  "files_written": ["/tmp/agent/output/filename.py"],
  "explanation": "what the code does",
  "iterations_to_fix": 0
}
"""


class CodingAgent(BaseAgent):

    agent_type    = "coding"
    capabilities  = [
        "python_execution",
        "error_driven_iteration",
        "file_read_write",
        "github_api",
        "code_generation",
        "debugging",
    ]
    default_tools  = ["code_executor", "filesystem", "api_caller"]
    system_prompt  = CODING_SYSTEM_PROMPT
    max_iterations = 8

    def _parse_output(self, raw: str) -> tuple[Any, float]:
        parsed, ok = try_parse_json(raw)
        if ok and parsed:
            confidence = extract_confidence(parsed)
            test_output = parsed.get("test_output", "")
            if test_output and "error" not in test_output.lower():
                confidence = min(1.0, confidence + 0.1)
            output = {
                "code":              parsed.get("code", ""),
                "test_output":       test_output,
                "files_written":     parsed.get("files_written", []),
                "explanation":       parsed.get("explanation", ""),
                "iterations_to_fix": parsed.get("iterations_to_fix", 0),
            }
            return output, confidence

        import re
        code_match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        code_str   = code_match.group(1).strip() if code_match else ""

        return {
            "code":              code_str,
            "test_output":       "",
            "files_written":     [],
            "explanation":       raw[:1000] if not code_str else "",
            "iterations_to_fix": 0,
        }, 0.5 if code_str else 0.3