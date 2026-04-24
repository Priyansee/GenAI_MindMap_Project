from agents.base_agent import BaseAgent
from typing import Any, Dict

class ExporterAgent(BaseAgent):
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__("Exporter Agent", api_key, model)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Export logic (PNG, PDF) is handled in the frontend using html2canvas and jsPDF.
        # This agent can provide the Miro-compatible JSON format.
        tree = input_data.get("mind_map", {})
        return self.to_json(tree)

    def to_json(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        # Simplified Miro-compatible schema
        nodes = []
        edges = []
        
        def traverse(node, parent_id=None):
            node_id = str(len(nodes))
            nodes.append({
                "id": node_id,
                "label": node.get("title", ""),
                "color": node.get("color", ""),
                "icon": node.get("icon", "")
            })
            if parent_id is not None:
                edges.append({"from": parent_id, "to": node_id})
            
            for child in node.get("children", []):
                traverse(child, node_id)
        
        traverse(tree)
        return {
            "type": "mindmap",
            "nodes": nodes,
            "edges": edges
        }
