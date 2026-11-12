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

def test_root_serves_html_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Render3D AI Studio" in response.text
    assert "canvas3d" in response.text

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

def test_get_asset_path_traversal_blocked():
    # Attempt to traverse outside storage/assets
    response = client.get("/api/v1/assets/../../api-key.txt")
    assert response.status_code in [404, 400]

def test_generate_image_to_3d_endpoint():
    # Upload dummy image
    image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
    files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
    data = {"caption": "Test character from reference image"}
    response = client.post("/api/v1/generate/image-to-3d", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "completed"
    assert "model_url" in res_data
    assert "qa_report" in res_data

def test_generate_character_neural_endpoint(monkeypatch):
    import trimesh
    from services.gateway import hunyuan_client

    def mock_gen_text(caption, output_path, steps=15, timeout=120):
        mesh = trimesh.creation.box(extents=(1, 2, 0.5))
        mesh.export(output_path)
        return output_path

    monkeypatch.setattr(hunyuan_client, "generate_3d_from_text", mock_gen_text)
    monkeypatch.setattr(hunyuan_client, "token", "mock_token")

    payload = {
        "prompt": "Realistic golden knight in plate armor",
        "style": "Game-Ready Realistic",
        "engine": "neural"
    }
    response = client.post("/api/v1/generate/character", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "completed"
    assert "Hunyuan3D-2" in res_data["engine"]
    assert res_data["qa_report"]["score"] >= 96


