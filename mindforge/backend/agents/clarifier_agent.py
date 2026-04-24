import json
from agents.base_agent import BaseAgent
from typing import Any, Dict

class ClarifierAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__("Clarifier Agent", api_key, model)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_input = input_data.get("input", "")
        
        prompt = f"""Analyze the user's initial idea: "{user_input}"

The input is currently vague. Generate 3-4 clarifying questions, ensuring at least one question offers different 'interpretations' of the user's intent.

Goals:
1. Provide 2-3 clear possible interpretations of the input as options.
2. Present them as clear MCQ options.
3. Ask the user to choose or refine.

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "id": 1,
      "question": "Which interpretation best fits your goal?",
      "options": ["Interpretation A...", "Interpretation B...", "Interpretation C..."]
    }},
    ...
  ]
}}

INPUT:
{user_input}"""

        raw_output = await self.call_llm(prompt)
        return self.parse_json(raw_output)
