import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app
import os

client = TestClient(app)

# System Black-Box Tests for MindForge API

def test_root_endpoint():
    """Verify the root endpoint is accessible and returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "MindForge API" in response.json()["message"]

def test_health_check():
    """Verify health check endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("main.ClarifierAgent")
def test_clarify_flow_standard_input(mock_clarifier_class):
    """Black-box test for /clarify with a standard valid input."""
    mock_clarifier = mock_clarifier_class.return_value
    mock_clarifier.run = AsyncMock(return_value={"questions": ["Q1", "Q2"]})

    payload = {"text": "What is quantum computing?", "api_key": "test_key"}
    response = client.post("/clarify", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "is_vague" in data
    assert "questions" in data
    assert len(data["questions"]) == 2

@patch("main.ClarifierAgent")
def test_clarify_vague_input_handling(mock_clarifier_class):
    """Verify that short inputs are flagged as vague."""
    mock_clarifier = mock_clarifier_class.return_value
    mock_clarifier.run = AsyncMock(return_value={"questions": ["Too short?"]})

    response = client.post("/clarify", json={"text": "AI", "api_key": "test"})
    assert response.status_code == 200
    assert response.json()["is_vague"] is True

def test_clarify_invalid_empty_input():
    """Verify that empty text returns a 400 error."""
    response = client.post("/clarify", json={"text": "", "api_key": "test"})
    assert response.status_code == 400
    assert "detail" in response.json()

@patch("main.PipelineController")
def test_generate_flow_standard_input(mock_controller_class):
    """Black-box test for /generate with valid input and answers."""
    mock_controller = mock_controller_class.return_value
    mock_controller.run = AsyncMock(return_value={
        "mind_map": {"title": "Quantum Computing", "nodes": []},
        "miro_json": {},
        "tester_report": {"quality_score": 0.95},
        "pipeline_log": []
    })

    payload = {
        "text": "Quantum Computing",
        "clarification_answers": {"Q1": "Yes"},
        "api_key": "test_key"
    }
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "mind_map" in data
    assert "tester_report" in data
    assert data["mind_map"]["title"] == "Quantum Computing"

def test_generate_missing_api_key():
    """Verify system rejects requests without API key (env or param)."""
    with patch.dict(os.environ, {}, clear=True):
        response = client.post("/generate", json={"text": "No Key"})
        assert response.status_code == 400
        assert "API key is required" in response.json()["detail"]

@patch("main.PipelineController")
def test_generate_pipeline_failure_response(mock_controller_class):
    """Verify that internal pipeline crashes are handled gracefully (500)."""
    mock_controller = mock_controller_class.return_value
    mock_controller.run = AsyncMock(side_effect=Exception("Critical Failure"))

    response = client.post("/generate", json={"text": "Fail", "api_key": "test"})
    assert response.status_code == 500
    assert "Pipeline error" in response.json()["detail"]

def test_generate_long_input_boundary():
    """Test system with a very large input string."""
    large_text = "word " * 500
    # Just checking request handling, not full execution
    with patch("main.PipelineController") as mock_controller:
        mock_controller.return_value.run = AsyncMock(return_value={"mind_map": {}})
        response = client.post("/generate", json={"text": large_text, "api_key": "test"})
        assert response.status_code == 200

def test_unsupported_method_on_endpoints():
    """Verify that unsupported HTTP methods return 405."""
    response = client.get("/generate")
    assert response.status_code == 405
    
    response = client.put("/clarify", json={"text": "test"})
    assert response.status_code == 405

def test_generate_response_schema():
    """Validate full response structure."""
    with patch("main.PipelineController") as mock_controller:
        mock_controller.return_value.run = AsyncMock(return_value={
            "mind_map": {"title": "AI", "nodes": []},
            "miro_json": {"data": []},
            "tester_report": {"quality_score": 0.9},
            "pipeline_log": ["step1", "step2"]
        })

        response = client.post("/generate", json={
            "text": "AI",
            "api_key": "test"
        })

        data = response.json()

        assert isinstance(data["mind_map"], dict)
        assert isinstance(data["miro_json"], dict)
        assert isinstance(data["tester_report"], dict)
        assert isinstance(data["pipeline_log"], list)

def test_invalid_json_payload():
    """System should reject malformed JSON."""
    response = client.post("/generate", data="invalid_json")
    assert response.status_code in [400, 422]

def test_missing_text_field():
    response = client.post("/generate", json={"api_key": "test"})
    assert response.status_code == 422

def test_missing_text_field():
    response = client.post("/generate", json={"api_key": "test"})
    assert response.status_code == 422

def test_invalid_data_types():
    response = client.post("/generate", json={
        "text": 12345,
        "api_key": "test"
    })
    assert response.status_code == 422

def test_generate_with_empty_answers():
    with patch("main.PipelineController") as mock_controller:
        mock_controller.return_value.run = AsyncMock(return_value={"mind_map": {}})

        response = client.post("/generate", json={
            "text": "AI",
            "clarification_answers": {},
            "api_key": "test"
        })

        assert response.status_code == 200

def test_extremely_large_input():
    large_text = "AI " * 5000

    with patch("main.PipelineController") as mock_controller:
        mock_controller.return_value.run = AsyncMock(return_value={"mind_map": {}})

        response = client.post("/generate", json={
            "text": large_text,
            "api_key": "test"
        })

        assert response.status_code == 200

def test_api_key_from_env():
    with patch.dict(os.environ, {"API_KEY": "env_key"}):
        with patch("main.PipelineController") as mock_controller:
            mock_controller.return_value.run = AsyncMock(return_value={"mind_map": {}})

            response = client.post("/generate", json={"text": "AI"})
            assert response.status_code == 200

@patch("main.PipelineController")
def test_pipeline_timeout(mock_controller_class):
    async def slow_run(*args, **kwargs):
        raise TimeoutError("Timeout")

    mock_controller_class.return_value.run = slow_run

    response = client.post("/generate", json={"text": "AI", "api_key": "test"})
    assert response.status_code == 500

def test_multiple_requests_stability():
    with patch("main.PipelineController") as mock_controller:
        mock_controller.return_value.run = AsyncMock(return_value={"mind_map": {}})

        for _ in range(5):
            response = client.post("/generate", json={
                "text": "AI",
                "api_key": "test"
            })
            assert response.status_code == 200