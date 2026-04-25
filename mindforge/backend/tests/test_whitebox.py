import pytest
import json
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from agents.base_agent import BaseAgent
from agents.validator_agent import ValidatorAgent
from pipeline_controller import PipelineController

# --- BaseAgent Tests ---

@pytest.mark.asyncio
async def test_base_agent_parse_json_variants():
    agent = BaseAgent("Test Agent", api_key="test")
    
    # 1. Clean JSON
    assert agent.parse_json('{"key": "value"}') == {"key": "value"}
    
    # 2. Markdown wrapped
    assert agent.parse_json('```json\n{"key": "value"}\n```') == {"key": "value"}
    
    # 3. Leading/Trailing text
    assert agent.parse_json('Here is the JSON: {"key": "value"} Hope it helps!') == {"key": "value"}
    
    # 4. Nested JSON
    assert agent.parse_json('{"outer": {"inner": 123}}') == {"outer": {"inner": 123}}

@pytest.mark.asyncio
async def test_base_agent_parse_json_failure():
    agent = BaseAgent("Test Agent", api_key="test")
    with pytest.raises(Exception):
        agent.parse_json('not a json at all')

@pytest.mark.asyncio
async def test_base_agent_call_llm_retry_on_429():
    agent = BaseAgent("Test Agent", api_key="test")
    
    # Mocking httpx.AsyncClient.post
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429
    
    mock_response_200 = MagicMock()
    mock_response_200.status_code = 200
    mock_response_200.json.return_value = {
        "choices": [{"message": {"content": '{"result": "success"}'}}]
    }
    
    # We want 2 failures then 1 success
    with patch("httpx.AsyncClient.post", side_effect=[mock_response_429, mock_response_429, mock_response_200]):
        # Reduce sleep time for testing
        with patch("asyncio.sleep", return_value=None):
            result = await agent.call_llm("test prompt")
            assert result == '{"result": "success"}'

@pytest.mark.asyncio
async def test_base_agent_missing_api_key():
    agent = BaseAgent("Test Agent", api_key=None)
    with patch("os.getenv", return_value=None):
        # We need to manually set api_key to None because __init__ might pick it up from env
        agent.api_key = None
        with pytest.raises(ValueError, match="API key missing"):
            await agent.call_llm("test")

# --- ValidatorAgent Tests ---

def test_validator_rule_check_logic():
    validator = ValidatorAgent(api_key="test")
    
    # 1. Valid tree
    valid_tree = {
        "title": "Root",
        "children": [
            {"title": "Child 1", "children": []},
            {"title": "Child 2", "children": []}
        ]
    }
    assert len(validator.rule_check(valid_tree)) == 0
    
    # 2. Duplicate titles
    dup_tree = {
        "title": "Root",
        "children": [
            {"title": "Same", "children": []},
            {"title": "Same", "children": []}
        ]
    }
    issues = validator.rule_check(dup_tree)
    assert any("Duplicate node titles" in issue for issue in issues)
    
    # 3. Max depth exceeded
    deep_tree = {
        "title": "1", "children": [
            {"title": "2", "children": [
                {"title": "3", "children": [
                    {"title": "4", "children": [
                        {"title": "5", "children": [
                            {"title": "6", "children": []}
                        ]}
                    ]}
                ]}
            ]}
        ]
    }
    issues = validator.rule_check(deep_tree)
    assert any("Maximum depth exceeded" in issue for issue in issues)
    
    # 4. Too many children
    wide_tree = {
        "title": "Root",
        "children": [{"title": f"C{i}", "children": []} for i in range(10)]
    }
    issues = validator.rule_check(wide_tree)
    assert any("more than 7 children" in issue for issue in issues)

# --- PipelineController Tests ---

@pytest.mark.asyncio
async def test_pipeline_controller_logging_edge_case():
    controller = PipelineController(api_key="test")
    controller._log_stage("TestAgent", "success", 0.1234, 100, 200)
    
    assert len(controller.logs) == 1
    log = controller.logs[0]
    assert log["agent_name"] == "TestAgent"
    assert log["duration_ms"] == 123
    assert log["input_length"] == 100
    assert log["output_length"] == 200
