from agents.base_agent import BaseAgent
from typing import Any, Dict

class PlannerAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__("Planner Agent", api_key, model)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_input = input_data.get("input", "")
        clarification = input_data.get("clarification_answers", {})
        additional = input_data.get("additional_info", "")
        
        clarification_text = "\n".join([f"Q: {k} A: {v}" for k, v in clarification.items()]) if clarification else "None"
        
        prompt = f"""Analyze the input and create a STRATEGIC ROADMAP.

CENTRAL GOAL: {user_input}
USER CLARIFICATIONS: {clarification_text}
SPECIFIC REQUIREMENTS: {additional}

Instructions for a logical "Path":
1. The Central Goal is the starting point on the far left.
2. Identify 4-5 Sequential Phases (e.g., Phase 1: Foundation, Phase 2: Execution, etc.).
3. Each phase must have a clear "Milestone" name.
4. Ensure the mapping follows a 'natural progression' from start to finish.
5. Branches must be distinct and not overlap in meaning.

Return ONLY valid JSON:
{{
  "central_topic": "...",
  "branches": ["...", "..."],
  "relationships": [
    {{ "from": "...", "to": "...", "label": "next step" }}
  ]
}}

INPUT:
{user_input}"""

        raw_output = await self.call_llm(prompt)
        return self.parse_json(raw_output)
