import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from services.gateway import app, orchestrator, ASSETS_DIR
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
def mock_orchestrator(monkeypatch, tmp_path):
    import services.gateway as gw
    monkeypatch.setattr(gw, "ASSETS_DIR", tmp_path)
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
        score=96,
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
    monkeypatch.setattr(orchestrator, "inspect_visual_alignment", MagicMock(return_value=mock_qa))
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
    assert "char_clean_" in data["model_url"]
    assert data.get("animations") == []
    assert "qa_report" in data
    assert data["qa_report"]["passed"] is True
    assert data["qa_report"]["score"] >= 90

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

def test_generate_image_to_3d_endpoint(monkeypatch, tmp_path):
    # Mock hunyuan_client generate_3d_from_image to verify successful neural route with texture colors matching reference
    import numpy as np
    import trimesh
    from services.gateway import hunyuan_client

    def mock_gen_img(image_path, output_path, caption="", steps=15, timeout=180, target_faces=25000, **kwargs):
        box = trimesh.creation.box(extents=(0.6, 1.75, 0.4))
        # Color matching reference red to pass Visual QA
        box.visual.vertex_colors = np.tile([220, 30, 30, 255], (len(box.vertices), 1))
        box.export(output_path)
        return output_path

    monkeypatch.setattr(hunyuan_client, "generate_3d_from_image", mock_gen_img)
    monkeypatch.setattr(hunyuan_client, "token", "mock_token")

    from PIL import Image
    import io
    pil_img = Image.new("RGB", (64, 64), (220, 30, 30))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    files = {"image": ("test.png", image_bytes, "image/png")}
    data = {"caption": "Test character from reference image"}
    response = client.post("/api/v1/generate/image-to-3d", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "completed"
    assert "Clay Geometry" in res_data["engine"]
    assert "model_url" in res_data
    assert "neural3d_" in res_data["model_url"]
    assert res_data.get("animations") == []
    assert "qa_report" in res_data
    assert res_data["qa_report"]["passed"] is True


def test_generate_image_to_3d_fallback_flags_warning(monkeypatch):
    from services.gateway import hunyuan_client
    monkeypatch.setattr(hunyuan_client, "token", None)

    from PIL import Image
    import io
    pil_img = Image.new("RGB", (64, 64), (220, 30, 30))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    files = {"image": ("test.png", image_bytes, "image/png")}
    data = {"caption": "Test character fallback"}
    response = client.post("/api/v1/generate/image-to-3d", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "warning_fallback"
    assert "GPU API Error" in res_data["engine"]
    assert res_data["qa_report"]["passed"] is False
    assert res_data["qa_report"]["score"] <= 60

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
    assert "char_clean_" in res_data["model_url"]
    assert res_data.get("animations") == []
    assert res_data["qa_report"]["score"] >= 96

def test_animate_character_endpoint(tmp_path):
    # First generate a test mesh in isolated tmp_path (ASSETS_DIR)
    import trimesh
    test_id = "animtest"
    test_file = tmp_path / f"char_clean_{test_id}.glb"
    box = trimesh.creation.box(extents=(0.6, 0.3, 1.75))
    box.export(str(test_file))

    payload = {
        "task_id": test_id,
        "animations": ["walk", "idle", "run", "attack"]
    }
    response = client.post("/api/v1/animate-character", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "model_url" in data
    assert "walk" in data["animations"]
    assert data["default_animation"] == "walk"

    # Also test alias /api/animate-character
    alias_resp = client.post("/api/animate-character", json=payload)
    assert alias_resp.status_code == 200


def test_animate_character_path_traversal_blocked():
    resp1 = client.post("/api/v1/animate-character", json={"task_id": "../../api-key.txt"})
    assert resp1.status_code == 400
    resp2 = client.post("/api/v1/animate-character", json={"model_url": "/api/v1/assets/../../api-key.txt"})
    assert resp2.status_code == 400


def test_empty_prompt_blocked_on_all_endpoints():
    # Empty character prompt blocked
    resp_char1 = client.post("/api/v1/generate/character", json={"prompt": ""})
    assert resp_char1.status_code == 400
    assert "Prompt không được để trống" in resp_char1.json()["detail"]

    resp_char2 = client.post("/api/v1/generate/character", json={"prompt": "   "})
    assert resp_char2.status_code == 400

    # Empty weapon prompt blocked
    resp_weap = client.post("/api/v1/generate/weapon", json={"prompt": "  "})
    assert resp_weap.status_code == 400

    # Empty map prompt blocked
    resp_map = client.post("/api/v1/generate/map", json={"prompt": ""})
    assert resp_map.status_code == 400

    # Empty image in image-to-3d blocked
    resp_img = client.post(
        "/api/v1/generate/image-to-3d",
        files={"image": ("empty.png", b"", "image/png")}
    )
    assert resp_img.status_code == 400




