import json
from agents.base_agent import BaseAgent
from typing import Any, Dict, List

class ValidatorAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__("Validator Agent", api_key, model)
        self.rules = [
            "No duplicate node titles at the same level",
            "No empty or null title fields",
            "Maximum depth of 4 levels",
            "Root node must exist and have a title",
            "Every node must have a children array",
            "Minimum 2 children on root node",
            "Maximum 7 children on any single node"
        ]

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        structured_output = input_data.get("structured_output", {})
        
        # In a real app, we might run rule_check first and only call LLM if issues found.
        # But per the design, it says "First applies rule-based checks, then uses LLM to fix any issues found."
        
        prompt = f"""You are a JSON validator and fixer.

Validation rules:
- No duplicate node titles at the same level
- No empty or null title fields — replace with 'Untitled'
- Maximum depth of 4 levels — truncate deeper nodes
- Root node must have a title
- Every node must have a children array (use [] if missing)
- Minimum 2 children on root
- Maximum 7 children per node — merge extras

Fix ALL issues and return the corrected JSON in the SAME schema format.
Return ONLY valid JSON with no markdown.

INPUT:
{json.dumps(structured_output, indent=2)}"""

        raw_output = await self.call_llm(prompt)
        return self.parse_json(raw_output)

    def rule_check(self, tree: Dict[str, Any]) -> List[str]:
        issues = []
        if not tree or "title" not in tree:
            issues.append("Root node must have a title")
        
        def traverse(node, depth, path):
            if depth > 4:
                issues.append(f"Maximum depth exceeded at {path}")
            
            if "children" not in node or not isinstance(node["children"], list):
                issues.append(f"Node '{node.get('title')}' missing children array")
                return
            
            titles = [c.get("title") for c in node["children"] if c.get("title")]
            if len(titles) != len(set(titles)):
                issues.append(f"Duplicate node titles at level {depth+1} under '{node.get('title')}'")
            
            if len(node["children"]) > 7:
                 issues.append(f"Node '{node.get('title')}' has more than 8 children")

            for child in node["children"]:
                traverse(child, depth + 1, path + " -> " + str(child.get("title")))

        if tree:
            traverse(tree, 1, str(tree.get("title")))
            if len(tree.get("children", [])) < 2:
                issues.append("Minimum 2 children on root node")

        return issues
