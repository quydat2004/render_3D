import pytest
from unittest.mock import MagicMock
from services.orchestrator.gemini_orchestrator import (
    GeminiOrchestrator,
    AssetType,
    OrthographicConceptPrompt,
    LevelLayoutSchema,
    QAInspectionResult
)

def test_orthographic_concept_prompt_schema():
    concept = OrthographicConceptPrompt(
        front_prompt="Front orthographic view of cyberpunk warrior",
        side_prompt="Side orthographic 90-degree view",
        back_prompt="Back orthographic view",
        style_notes="Flat neutral diffuse lighting, no shadows",
        pose="A-pose",
        poly_budget=25000
    )
    assert concept.pose == "A-pose"
    assert concept.poly_budget == 25000

def test_level_layout_schema():
    layout = LevelLayoutSchema(
        level_id="cyber_arena_01",
        dimensions=[40.0, 60.0],
        grid_size_m=2.0,
        biome="Industrial_SciFi",
        rooms=[{"id": "r1", "bounds": [0, 0, 20, 20]}],
        corridors=[{"connects": ["r1"], "width": 4.0}],
        spawn_points=[{"team": "alpha", "pos": [2, 0, 2]}],
        instances=[{
            "asset_id": "wall_concrete_4x4",
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0]
        }]
    )
    assert layout.level_id == "cyber_arena_01"
    assert layout.dimensions == [40.0, 60.0]
    assert len(layout.instances) == 1

def test_qa_inspection_result_schema():
    qa_pass = QAInspectionResult(
        score=92,
        passed=True,
        anomalies_detected=[],
        corrective_instructions="None. Model is game-ready."
    )
    assert qa_pass.passed is True

def test_orchestrator_offline_mode_prompt_expansion():
    # When offline_mode=True, orchestrator generates deterministic high-quality schema without remote API call
    orchestrator = GeminiOrchestrator(offline_mode=True)
    res = orchestrator.expand_asset_prompt("Cyberpunk Katana", AssetType.WEAPON)
    assert isinstance(res, OrthographicConceptPrompt)
    assert res.poly_budget == 8000
    assert "katana" in res.front_prompt.lower()

def test_orchestrator_offline_mode_plan_environment():
    orchestrator = GeminiOrchestrator(offline_mode=True)
    layout = orchestrator.plan_environment("Industrial sci-fi warehouse")
    assert isinstance(layout, LevelLayoutSchema)
    assert layout.grid_size_m == 2.0
    assert len(layout.instances) > 0

def test_orchestrator_offline_mode_inspect_turntable():
    orchestrator = GeminiOrchestrator(offline_mode=True)
    qa = orchestrator.inspect_turntable([b"frame_1", b"frame_2"], AssetType.CHARACTER)
    assert isinstance(qa, QAInspectionResult)
    assert qa.score >= 85
    assert qa.passed is True

def test_orchestrator_expand_asset_prompt_with_mock():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '''{
        "front_prompt": "Front orthographic view of scifi rifle",
        "side_prompt": "Side profile showing stock and barrel",
        "back_prompt": "Rear view with socket alignment",
        "style_notes": "Hard-surface metal and matte polymer",
        "pose": "Neutral",
        "poly_budget": 8000
    }'''
    mock_client.models.generate_content.return_value = mock_response

    orchestrator = GeminiOrchestrator(api_key="dummy_key", client=mock_client)
    result = orchestrator.expand_asset_prompt(
        user_prompt="Futuristic assault rifle",
        asset_type=AssetType.WEAPON
    )

    assert isinstance(result, OrthographicConceptPrompt)
    assert result.poly_budget == 8000
    assert "rifle" in result.front_prompt.lower()

def test_orchestrator_inspect_visual_alignment_detects_planar_projection(tmp_path):
    import numpy as np
    import trimesh
    from PIL import Image

    orchestrator = GeminiOrchestrator(offline_mode=True)
    box = trimesh.creation.box(extents=(0.6, 1.8, 0.4))
    # Planar projection: every vertex, front and back, maps to the same UV band.
    uvs = np.zeros((len(box.vertices), 2), dtype=np.float32)
    uvs[:, 0] = 0.5
    tex_img = Image.new("RGB", (64, 64), (200, 30, 30))
    box.visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_img)
    bad_glb = str(tmp_path / "planar.glb")
    box.export(bad_glb)

    qa = orchestrator.inspect_visual_alignment(bad_glb)
    assert qa.passed is False
    assert any("planar" in a.lower() for a in qa.anomalies_detected)
    assert qa.score <= 50

def test_orchestrator_inspect_visual_alignment_passes_valid_model(tmp_path):
    import numpy as np
    import trimesh
    from PIL import Image

    orchestrator = GeminiOrchestrator(offline_mode=True)
    box = trimesh.creation.box(extents=(0.6, 1.8, 0.4))
    verts = box.vertices
    # A real atlas: front face and back face live in different UV islands.
    uvs = np.zeros((len(verts), 2), dtype=np.float32)
    uvs[:, 0] = np.where(verts[:, 2] > 0, 0.25, 0.75)
    tex_img = Image.new("RGB", (64, 64), (200, 30, 30))
    box.visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_img)
    good_glb = str(tmp_path / "valid.glb")
    box.export(good_glb)

    qa = orchestrator.inspect_visual_alignment(good_glb)
    assert qa.passed is True
    assert qa.score >= 85
    assert len(qa.anomalies_detected) == 0

