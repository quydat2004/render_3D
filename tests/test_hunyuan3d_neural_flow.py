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

def test_no_planar_projection_fallback(tmp_path):
    """Paint down: mesh gets a flat vertex tint, never the source photo projected through the mesh."""
    from PIL import Image
    client = Hunyuan3DClient()
    box = trimesh.creation.box(extents=(1.0, 2.0, 0.5))
    raw_glb = str(tmp_path / "raw.glb")
    box.export(raw_glb)

    img = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
    img.paste(Image.new("RGB", (32, 32), (220, 30, 30)), (16, 16))
    img_path = str(tmp_path / "ref.png")
    img.save(img_path)

    out_glb = str(tmp_path / "tinted.glb")
    client._flat_color_fallback(raw_glb, img_path, out_glb)

    scene = trimesh.load(out_glb)
    geom = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    assert not (hasattr(geom.visual, "uv") and geom.visual.uv is not None and len(geom.visual.uv))
    assert not hasattr(client, "project_texture_to_mesh")
