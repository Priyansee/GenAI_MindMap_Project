import pytest
from unittest.mock import AsyncMock, patch
from mindforge.backend.pipeline_controller import PipelineController

@pytest.mark.asyncio
async def test_full_pipeline_success():
    # Mock responses for all agents
    planner_res = '{"central_topic": "Test", "branches": ["B1"], "relationships": []}'
    structurer_res = '{"title": "Test", "children": [{"title": "B1", "children": []}]}'
    validator_res = structurer_res
    enhancer_res = '{"title": "Test", "color": "#c8a96e", "children": [{"title": "B1", "color": "#7eb8a6", "children": []}]}'
    tester_res = '{"valid": true, "quality_score": 0.9, "issues": []}'

    with patch("mindforge.backend.agents.base_agent.BaseAgent.call_llm", side_effect=[
        planner_res, structurer_res, validator_res, enhancer_res, tester_res
    ]):
        controller = PipelineController(api_key="test_key")
        result = await controller.run("Test input")
        
        assert "mind_map" in result
        assert "tester_report" in result
        assert "pipeline_log" in result
        assert len(result["pipeline_log"]) == 7  # All 7 agents should have logged
        assert result["tester_report"]["quality_score"] == 0.9

@pytest.mark.asyncio
async def test_pipeline_graceful_degradation():
    # Simulate enhancer failure
    planner_res = '{"central_topic": "Test", "branches": ["B1"], "relationships": []}'
    structurer_res = '{"title": "Test", "children": [{"title": "B1", "children": []}]}'
    validator_res = structurer_res
    tester_res = '{"valid": true, "quality_score": 0.5, "issues": ["Enhancer failed"]}'

    with patch("mindforge.backend.agents.base_agent.BaseAgent.call_llm", side_effect=[
        planner_res, structurer_res, validator_res, Exception("Enhancer Error"), tester_res
    ]):
        controller = PipelineController(api_key="test_key")
        result = await controller.run("Test input")
        
        assert result["mind_map"]["title"] == "Test"
        assert result["pipeline_log"][3]["status"] == "skipped"  # Enhancer (index 3)
        assert result["pipeline_log"][3]["agent_name"] == "Enhancer Agent"
