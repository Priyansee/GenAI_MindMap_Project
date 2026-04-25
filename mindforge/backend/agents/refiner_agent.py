import json
from agents.base_agent import BaseAgent
from typing import Any, Dict

class RefinerAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__("Refiner Agent", api_key, model)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        previous_map = input_data.get("previous_map", {})
        feedback = input_data.get("feedback", "")
        
        prompt = f"""You are a Refinement Specialist. 
Your task is to update an existing Mind Map based on user feedback.

CURRENT MIND MAP:
{json.dumps(previous_map, indent=2)}

USER FEEDBACK:
"{feedback}"

INSTRUCTIONS:
1. Modify the mind map to incorporate the user's feedback.
2. Maintain the overall hierarchical structure.
3. Keep the same JSON schema (title, summary, flow_explanation, children with title/icon/color).
4. Do NOT change parts of the map that the user didn't mention, unless necessary for consistency.

Return ONLY valid JSON:
{{
  "title": "...",
  "summary": "...",
  "flow_explanation": "...",
  "children": [...]
}}"""

        raw_output = await self.call_llm(prompt)
        return self.parse_json(raw_output)
