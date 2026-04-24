import json
from agents.base_agent import BaseAgent
from typing import Any, Dict, List

class TesterAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__("Tester Agent", api_key, model)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pipeline_outputs = input_data.get("pipeline_outputs", {})
        
        prompt = f"""Verify the pipeline output quality.

Checks:
✔ Node count consistency between stages
✔ Central topic preserved from input to output
✔ No node exceeds depth 4
✔ All titles are non-empty strings
✔ Structure is valid hierarchical JSON

Return ONLY valid JSON:
{{
  "valid": true,
  "issues": [],
  "quality_score": 0.9,
  "agent_statuses": {{}}
}}

INPUT:
{json.dumps(pipeline_outputs, indent=2)}"""

        raw_output = await self.call_llm(prompt)
        return self.parse_json(raw_output)
