import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import os
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to MindForge API" in response.json()["message"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("main.ClarifierAgent")
def test_clarify_vague_input(mock_clarifier_class):
    mock_clarifier = mock_clarifier_class.return_value
    mock_clarifier.run = AsyncMock(return_value={"questions": ["What do you mean?"]})

    response = client.post("/clarify", json={"text": "vague", "api_key": "test_key"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_vague"] is True
    assert data["word_count"] == 1
    assert data["questions"] == ["What do you mean?"]

@patch("main.ClarifierAgent")
def test_clarify_detailed_input(mock_clarifier_class):
    mock_clarifier = mock_clarifier_class.return_value
    mock_clarifier.run = AsyncMock(return_value={"questions": []})

    long_text = "this is a very specific sentence with more than five words"
    response = client.post("/clarify", json={"text": long_text, "api_key": "test_key"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_vague"] is False
    assert data["word_count"] == len(long_text.split())
    assert data["questions"] == []

@patch("main.ClarifierAgent")
def test_clarify_boundary_10_words(mock_clarifier_class):
    mock_clarifier = mock_clarifier_class.return_value
    mock_clarifier.run = AsyncMock(return_value={"questions": []})

    ten_word_text = "one two three four five six seven eight nine ten"
    response = client.post("/clarify", json={"text": ten_word_text, "api_key": "test_key"})
    assert response.status_code == 200
    data = response.json()
    # is_vague = word_count < 10. So 10 words should NOT be vague.
    assert data["is_vague"] is False
    assert data["word_count"] == 10

def test_clarify_no_text():
    response = client.post("/clarify", json={"text": ""})
    assert response.status_code == 400

@patch.dict(os.environ, {}, clear=True)
def test_generate_mind_map_no_api_key():
    response = client.post("/generate", json={"text": "test_text"})
    assert response.status_code == 400
    assert "API key is required" in response.json()["detail"]

def test_generate_mind_map_no_text():
    response = client.post("/generate", json={"text": "", "api_key": "test"})
    assert response.status_code == 400
    assert "Text input is required" in response.json()["detail"]

@patch("main.PipelineController")
def test_generate_mind_map_success(mock_controller_class):
    mock_controller = mock_controller_class.return_value
    mock_controller.run = AsyncMock(return_value={
        "mind_map": "mapped_data",
        "tester_report": {"valid": True}
    })

    response = client.post("/generate", json={"text": "create a map about AI", "api_key": "test_key"})
    assert response.status_code == 200
    assert response.json()["mind_map"] == "mapped_data"

@patch("main.PipelineController")
def test_generate_mind_map_controller_failure(mock_controller_class):
    mock_controller = mock_controller_class.return_value
    mock_controller.run = AsyncMock(side_effect=Exception("Simulated error"))

    response = client.post("/generate", json={"text": "create a map about AI", "api_key": "test_key"})
    assert response.status_code == 500
    assert "Pipeline error: Simulated error" in response.json()["detail"]

@patch("main.ClarifierAgent")
def test_clarify_exception(mock_clarifier_class):
    mock_clarifier = mock_clarifier_class.return_value
    mock_clarifier.run = AsyncMock(side_effect=Exception("Clarification Error"))

    response = client.post("/clarify", json={"text": "test", "api_key": "test_key"})
    assert response.status_code == 500
    assert "Clarification failed: Clarification Error" in response.json()["detail"]
