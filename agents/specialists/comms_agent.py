from __future__ import annotations

from typing import Any

from agents.base import BaseAgent, try_parse_json

COMMS_SYSTEM_PROMPT = """\
You are a Communication Agent. Your tool is api_caller.

You interact with email and calendar APIs. ALL state-changing actions
(POST, PUT, DELETE) require explicit human approval before you proceed.

Your workflow for every task:
1. PLAN: describe exactly what you are about to do (what email to send,
   to whom, what calendar event to create).
2. DRAFT: write the complete email body or event description.
3. WAIT: state "I am waiting for approval to [action]. Please confirm."
   Then use api_caller with method GET to check approval status, OR
   wait for the observation to contain "APPROVED" or "REJECTED".
4. EXECUTE: only after you see "APPROVED" in observations, make the
   state-changing API call.
5. CONFIRM: verify the action succeeded by checking the API response.

CRITICAL RULES:
- Never send an email or create a calendar event without approval.
- Always show the complete draft before sending.
- If approval is rejected: acknowledge it and report what was NOT done.
- Use professional language. Proofread before sending.
- For emails: Subject must be clear and specific (not "Update" or "Info").
- For calendar: Include duration, timezone, and description in every event.

You MUST end with: FINAL ANSWER:
Then output either JSON or a plain confirmation:
{
  "action_taken": "sent email / created event / rejected",
  "recipient":    "email or calendar name",
  "subject":      "email subject or event title",
  "approved_by":  "human",
  "status":       "sent / created / rejected / pending_approval",
  "api_response": "HTTP status and key fields from the API response"
}
"""


class CommsAgent(BaseAgent):
    agent_type    = "communication"
    capabilities  = [
        "email_send",
        "email_draft",
        "calendar_create",
        "calendar_update",
        "hitl_approval",
        "professional_formatting",
    ]
    default_tools  = ["api_caller"]
    system_prompt  = COMMS_SYSTEM_PROMPT
    max_iterations = 4

    def _parse_output(self, raw: str) -> tuple[Any, float]:
        parsed, ok = try_parse_json(raw)
        if ok and parsed:
            status = parsed.get("status", "").lower()
            confidence = {
                "sent":             0.95,
                "created":          0.95,
                "rejected":         0.70,
                "pending_approval": 0.50,
            }.get(status, 0.80)

            output = {
                "action_taken": parsed.get("action_taken", ""),
                "recipient":    parsed.get("recipient", ""),
                "subject":      parsed.get("subject", ""),
                "approved_by":  parsed.get("approved_by", ""),
                "status":       status,
                "api_response": parsed.get("api_response", ""),
            }
            return output, confidence

        lower = raw.lower()
        if any(w in lower for w in ("sent", "created", "delivered", "confirmed")):
            conf = 0.75
        elif any(w in lower for w in ("rejected", "denied", "not sent")):
            conf = 0.70
        else:
            conf = 0.40

        return {
            "action_taken": raw[:500] if raw else "No confirmation produced.",
            "recipient":    "",
            "subject":      "",
            "approved_by":  "",
            "status":       "unknown",
            "api_response": "",
        }, conf