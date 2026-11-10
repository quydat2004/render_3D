import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from services.gateway import app

client = TestClient(app)

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
