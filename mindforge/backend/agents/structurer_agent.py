import json
from agents.base_agent import BaseAgent
from typing import Any, Dict

class StructurerAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__("Structurer Agent", api_key, model)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        planner_output = input_data.get("planner_output", {})
        
        prompt = f"""Convert the planner output into a professional, deep hierarchical JSON tree.

Rules for high-quality structure:
- 'central_topic' is the Root.
- 'branches' are the level 1 children.
- You MUST create logical sub-topics (level 2 and 3) for each branch based on common knowledge.
- Keep titles extremely concise (max 3 words).
- Ensure a balanced tree structure.
- Every node MUST have a 'title' and a 'children' array.

Return ONLY valid JSON:
{{
  "title": "...",
  "children": [
    {{
      "title": "...",
      "children": []
    }}
  ]
}}

INPUT:
{json.dumps(planner_output, indent=2)}"""

        raw_output = await self.call_llm(prompt)
        return self.parse_json(raw_output)
