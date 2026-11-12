import pytest
from pathlib import Path
import trimesh
from services.inference.hunyuan3d_client import Hunyuan3DClient

def test_concept_image_generation(tmp_path):
    client = Hunyuan3DClient()
    out_img = str(tmp_path / "concept.png")
    res_path = client.generate_concept_image("realistic golden knight in armor", out_img)
    assert Path(res_path).exists()
    assert Path(res_path).stat().st_size > 500

def test_project_texture_to_mesh_creates_material(tmp_path):
    client = Hunyuan3DClient()
    # Create simple box mesh
    box = trimesh.creation.box(extents=(1.0, 2.0, 0.5))
    raw_glb = str(tmp_path / "raw.glb")
    box.export(raw_glb)

    # Generate test image
    img_path = str(tmp_path / "concept.png")
    client.generate_concept_image("knight", img_path)

    out_glb = str(tmp_path / "textured.glb")
    res_path = client.project_texture_to_mesh(raw_glb, img_path, out_glb)

    assert Path(res_path).exists()
    scene = trimesh.load(res_path)
    geom = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    assert hasattr(geom.visual, "uv") and geom.visual.uv is not None
    assert len(geom.visual.uv) == len(geom.vertices)
