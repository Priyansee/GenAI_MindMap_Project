import pytest
import asyncio
import time
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from main import app
from fastapi.testclient import TestClient
from pipeline_controller import PipelineController
from agents.base_agent import BaseAgent

client = TestClient(app)

# --- NFR1: Vague check within 200ms ---
@pytest.mark.asyncio
async def test_nfr1_vague_check_latency():
    # Use a real string but mock the ClarifierAgent to isolate the controller logic
    with patch("main.ClarifierAgent") as mock_clarifier:
        mock_clarifier.return_value.run = AsyncMock(return_value={"questions": []})
        
        start_time = time.perf_counter()
        response = client.post("/clarify", json={"text": "This is a test sentence that is quite long to check word count logic.", "api_key": "test"})
        duration = (time.perf_counter() - start_time) * 1000 # ms
        
        assert response.status_code == 200
        assert duration < 200, f"Vague check took {duration:.2f}ms, exceeds 200ms limit"

# --- NFR2: Pipeline completion within 10-15s ---
@pytest.mark.asyncio
async def test_nfr2_pipeline_latency():
    # We mock the LLM calls to take ~1.5s each (1.5 * 5 agents = 7.5s total + overhead)
    # This verifies the controller doesn't add more than a few seconds of overhead.
    
    async def slow_mock(*args, **kwargs):
        await asyncio.sleep(1.5)
        return '{"result": "success"}'

    with patch("agents.base_agent.BaseAgent.call_llm", side_effect=slow_mock):
        controller = PipelineController(api_key="test")
        start_time = time.perf_counter()
        await controller.run("Test input")
        duration = time.perf_counter() - start_time
        
        assert 10 <= duration <= 16, f"Pipeline took {duration:.2f}s, expected 10-15s range"

# --- NFR3: 10 simultaneous users ---
@pytest.mark.asyncio
async def test_nfr3_concurrency_10_users():
    # Mock LLM to return immediately to test system overhead under load
    with patch("agents.base_agent.BaseAgent.call_llm", AsyncMock(return_value='{"status": "ok"}')):
        controller = PipelineController(api_key="test")
        
        start_time = time.perf_counter()
        # Run 10 requests concurrently
        tasks = [controller.run(f"Input {i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start_time
        
        assert len(results) == 10
        # If speed is compromised, 10 concurrent requests might take much longer than 1 request
        # Since we mocked them to be instant, this should be very fast (< 1s overhead)
        assert duration < 5.0, f"Concurrency test took {duration:.2f}s, expected low overhead"

# --- NFR11: Pluggable Architecture ---
def test_nfr11_agent_inheritance():
    from agents.planner_agent import PlannerAgent
    from agents.structurer_agent import StructurerAgent
    from agents.validator_agent import ValidatorAgent
    
    # Verify all agents inherit from BaseAgent
    assert issubclass(PlannerAgent, BaseAgent)
    assert issubclass(StructurerAgent, BaseAgent)
    assert issubclass(ValidatorAgent, BaseAgent)
    
    # Verify they use the base_url but could be overridden
    agent = PlannerAgent(api_key="test")
    assert agent.base_url == "https://api.groq.com/openai/v1"

# --- NFR5 & NFR6 are already covered in test_whitebox.py and test_pipeline_full.py ---
# But we can add a quick check for NFR5 retry again here if needed.
@pytest.mark.asyncio
async def test_nfr5_retry_mechanism_verified():
    agent = BaseAgent("Test", api_key="test")
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429
    
    mock_response_200 = MagicMock()
    mock_response_200.status_code = 200
    mock_response_200.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    
    with patch("httpx.AsyncClient.post", side_effect=[mock_response_429, mock_response_200]):
        with patch("asyncio.sleep", return_value=None):
            await agent.call_llm("test")
            # If it didn't raise, retry worked
