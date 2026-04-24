import pytest
from unittest.mock import AsyncMock, patch
from mindforge.backend.pipeline_controller import PipelineController

@pytest.mark.asyncio
async def test_pipeline_sprint1_success():
    # Mock data for PlannerAgent
    planner_mock_response = """
    {
      "central_topic": "AI Startups",
      "branches": ["Idea", "Funding", "Team", "Product"],
      "relationships": []
    }
    """
    
    # Mock data for StructurerAgent
    structurer_mock_response = """
    {
      "title": "AI Startups",
      "children": [
        {"title": "Idea", "children": []},
        {"title": "Funding", "children": []},
        {"title": "Team", "children": []},
        {"title": "Product", "children": []}
      ]
    }
    """

    with patch("mindforge.backend.agents.base_agent.BaseAgent.call_llm", side_effect=[planner_mock_response, structurer_mock_response]):
        controller = PipelineController(api_key="test_key")
        result = await controller.run("Plan an AI startup")
        
        assert "mind_map" in result
        assert result["mind_map"]["title"] == "AI Startups"
        assert len(result["mind_map"]["children"]) == 4
        assert len(result["pipeline_log"]) == 2
        assert result["pipeline_log"][0]["agent_name"] == "Planner Agent"
        assert result["pipeline_log"][0]["status"] == "success"
        assert result["pipeline_log"][1]["agent_name"] == "Structurer Agent"
        assert result["pipeline_log"][1]["status"] == "success"

@pytest.mark.asyncio
async def test_pipeline_planner_failure():
    with patch("mindforge.backend.agents.base_agent.BaseAgent.call_llm", side_effect=Exception("API Error")):
        controller = PipelineController(api_key="test_key")
        with pytest.raises(Exception) as excinfo:
            await controller.run("Plan an AI startup")
        assert "API Error" in str(excinfo.value)
