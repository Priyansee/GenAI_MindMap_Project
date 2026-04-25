import pytest
from unittest.mock import AsyncMock, patch
from pipeline_controller import PipelineController

@pytest.fixture
def mock_agents():
    with patch("pipeline_controller.PlannerAgent") as p_planner, \
         patch("pipeline_controller.StructurerAgent") as p_structurer, \
         patch("pipeline_controller.ValidatorAgent") as p_validator, \
         patch("pipeline_controller.EnhancerAgent") as p_enhancer, \
         patch("pipeline_controller.RendererAgent") as p_renderer, \
         patch("pipeline_controller.ExporterAgent") as p_exporter, \
         patch("pipeline_controller.TesterAgent") as p_tester:
         
        p_planner.return_value.run = AsyncMock(return_value={"plan": "test"})
        p_structurer.return_value.run = AsyncMock(return_value={"struct": "test"})
        p_validator.return_value.run = AsyncMock(return_value={"valid": "test"})
        p_enhancer.return_value.run = AsyncMock(return_value={"enhance": "test"})
        p_renderer.return_value.run = AsyncMock(return_value={"render": "test"})
        p_exporter.return_value.run = AsyncMock(return_value={"export": "test"})
        p_tester.return_value.run = AsyncMock(return_value={"quality_score": 10.0})
         
        yield {
            "planner": p_planner,
            "structurer": p_structurer,
            "validator": p_validator,
            "enhancer": p_enhancer,
            "renderer": p_renderer,
            "exporter": p_exporter,
            "tester": p_tester,
        }

@pytest.mark.asyncio
async def test_pipeline_run_success(mock_agents):
    controller = PipelineController(api_key="test_key")
    result = await controller.run("test input")
    
    assert "mind_map" in result
    assert "tester_report" in result
    assert "pipeline_log" in result
    
    logs = result["pipeline_log"]
    assert len(logs) == 7  # All 7 agents succeeded
    assert logs[0]["agent_name"] == "Planner Agent"
    assert logs[0]["status"] == "success"

@pytest.mark.asyncio
async def test_pipeline_planner_fails(mock_agents):
    mock_agents["planner"].return_value.run.side_effect = Exception("Planner error")
    
    controller = PipelineController(api_key="test_key")
    with pytest.raises(Exception, match="Planner error"):
        await controller.run("test input")
    
    logs = controller.logs
    assert len(logs) == 1
    assert logs[0]["status"] == "failed"

@pytest.mark.asyncio
async def test_pipeline_structurer_retry(mock_agents):
    # structurer fails first time, succeeds second time
    mock_run = AsyncMock(side_effect=[Exception("First error"), {"struct": "retry_success"}])
    mock_agents["structurer"].return_value.run = mock_run
    
    controller = PipelineController(api_key="test_key")
    result = await controller.run("test input")
    
    assert mock_run.call_count == 2
    logs = result["pipeline_log"]
    # Check that a retry was logged
    structurer_logs = [l for l in logs if l["agent_name"] == "Structurer Agent"]
    assert structurer_logs[0]["status"] == "retried"

@pytest.mark.asyncio
async def test_pipeline_validator_skip_on_error(mock_agents):
    mock_agents["validator"].return_value.run.side_effect = Exception("Validator error")
    
    controller = PipelineController(api_key="test_key")
    result = await controller.run("test input")
    
    logs = result["pipeline_log"]
    validator_logs = [l for l in logs if l["agent_name"] == "Validator Agent"]
    assert validator_logs[0]["status"] == "skipped"
    # Even if validator skipped, the pipeline should complete successfully
    assert len(logs) == 7

@pytest.mark.asyncio
async def test_pipeline_structurer_total_failure(mock_agents):
    # structurer fails both times
    mock_run = AsyncMock(side_effect=[Exception("First error"), Exception("Second error")])
    mock_agents["structurer"].return_value.run = mock_run
    
    controller = PipelineController(api_key="test_key")
    with pytest.raises(Exception, match="Second error"):
        await controller.run("test input")

@pytest.mark.asyncio
async def test_pipeline_renderer_fails(mock_agents):
    mock_agents["renderer"].return_value.run.side_effect = Exception("Renderer error")
    
    controller = PipelineController(api_key="test_key")
    result = await controller.run("test input")
    
    logs = result["pipeline_log"]
    renderer_logs = [l for l in logs if l["agent_name"] == "Renderer Agent"]
    assert renderer_logs[0]["status"] == "failed"
    assert result["mind_map"] is not None # Falls back to previous stage output

@pytest.mark.asyncio
async def test_pipeline_exporter_fails(mock_agents):
    mock_agents["exporter"].return_value.run.side_effect = Exception("Exporter error")
    
    controller = PipelineController(api_key="test_key")
    result = await controller.run("test input")
    
    logs = result["pipeline_log"]
    exporter_logs = [l for l in logs if l["agent_name"] == "Exporter Agent"]
    assert exporter_logs[0]["status"] == "failed"
    assert result["miro_json"] is None

@pytest.mark.asyncio
async def test_pipeline_tester_fails(mock_agents):
    mock_agents["tester"].return_value.run.side_effect = Exception("Tester error")
    
    controller = PipelineController(api_key="test_key")
    result = await controller.run("test input")
    
    logs = result["pipeline_log"]
    tester_logs = [l for l in logs if l["agent_name"] == "Tester Agent"]
    assert tester_logs[0]["status"] == "failed"
    assert result["tester_report"]["quality_score"] == 0.0
