import json
from agents.base_agent import BaseAgent
from typing import Any, Dict

class EnhancerAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__("Enhancer Agent", api_key, model)
        self.color_palette = [
            "#c8a96e", "#7eb8a6", "#a47eb8", "#7e9eb8", 
            "#b87e7e", "#7eb87e", "#b8a87e", "#6ea8c8"
        ]

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated_output = input_data.get("validated_output", {})
        
        prompt = f"""Enhance the following roadmap structure with professional visual details and a narrative explanation.

ROADMAP: {json.dumps(validated_output)}

Tasks:
1. Add relevant emojis/icons to every node.
2. Assign a 'color' (hex code) from the provided palette to each node.
3. Write a concise 'summary' (2-3 sentences) of the overall strategy.
4. Write a 'flow_explanation' (bullet points) describing the logical progression.

Return ONLY valid JSON:
{{
  "title": "...",
  "summary": "...",
  "flow_explanation": "...",
  "children": [
    {{
      "title": "...",
      "icon": "...",
      "color": "...",
      "tag": "...",
      "children": [...]
    }}
  ]
}}"""

        raw_output = await self.call_llm(prompt)
        return self.parse_json(raw_output)
