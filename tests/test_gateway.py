import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from services.gateway import app, orchestrator
from services.orchestrator.gemini_orchestrator import (
    OrthographicConceptPrompt,
    LevelLayoutSchema,
    QAInspectionResult,
    RoomSchema,
    CorridorSchema,
    SpawnPointSchema,
    InstanceSchema
)

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_orchestrator(monkeypatch):
    # Mock orchestrator methods to prevent slow remote image network uploads during test runs
    mock_concept = OrthographicConceptPrompt(
        front_prompt="Front test view",
        side_prompt="Side test view",
        back_prompt="Back test view",
        style_notes="Clean lighting",
        pose="A-pose",
        poly_budget=20000
    )
    mock_qa = QAInspectionResult(
        score=95,
        passed=True,
        anomalies_detected=[],
        corrective_instructions="Geometry and textures verified."
    )
    mock_layout = LevelLayoutSchema(
        level_id="test_subway_arena",
        dimensions=[40.0, 40.0],
        grid_size_m=2.0,
        biome="Industrial_SciFi",
        rooms=[RoomSchema(id="room_1", bounds=[0, 0, 20, 20])],
        corridors=[CorridorSchema(connects=["room_1"], width=4.0)],
        spawn_points=[SpawnPointSchema(team="alpha", pos=[2.0, 0.0, 2.0])],
        instances=[InstanceSchema(asset_id="wall_4x4", position=[0.0, 0.0, 0.0])]
    )

    monkeypatch.setattr(orchestrator, "expand_asset_prompt", MagicMock(return_value=mock_concept))
    monkeypatch.setattr(orchestrator, "inspect_turntable", MagicMock(return_value=mock_qa))
    monkeypatch.setattr(orchestrator, "plan_environment", MagicMock(return_value=mock_layout))

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "blender_available" in data

def test_generate_character_endpoint(tmp_path):
    payload = {
        "prompt": "Cyberpunk female ninja",
        "style": "Sci-Fi Realistic",
        "poly_budget": 20000
    }
    response = client.post("/api/v1/generate/character", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["asset_type"] == "character"
    assert "model_url" in data
    assert "qa_report" in data
    assert data["qa_report"]["passed"] is True
    assert data["qa_report"]["score"] == 95

def test_generate_weapon_endpoint():
    payload = {
        "prompt": "Laser sniper rifle",
        "weapon_type": "ranged",
        "poly_budget": 8000
    }
    response = client.post("/api/v1/generate/weapon", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["asset_type"] == "weapon"
    assert len(data["sockets"]) >= 3
    assert any(s["name"] == "SOCKET_Muzzle" for s in data["sockets"])
    assert data["qa_report"]["passed"] is True

def test_generate_map_endpoint():
    payload = {
        "prompt": "Cyberpunk back alley arena",
        "dimensions": [40.0, 40.0]
    }
    response = client.post("/api/v1/generate/map", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["asset_type"] == "environment"
    assert "navmesh" in data
    assert len(data["classified_props"]) > 0
