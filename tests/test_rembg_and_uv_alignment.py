import pytest
import numpy as np
from PIL import Image, ImageDraw
import trimesh
from pathlib import Path
from services.inference.hunyuan3d_client import Hunyuan3DClient

def test_remove_background_cleans_white_padding(tmp_path):
    client = Hunyuan3DClient()
    # Create an image with white background and a centered red character shape
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Character body in center (x: 150..250, y: 100..320)
    draw.rectangle([150, 100, 250, 320], fill=(220, 20, 20))
    raw_path = str(tmp_path / "raw_with_bg.png")
    img.save(raw_path)

    clean_img = client.remove_background(img)
    arr = np.array(clean_img)
    assert clean_img.mode == "RGBA"
    # Corner should be transparent
    assert arr[0, 0, 3] == 0
    # Center character body should be fully opaque
    assert arr[200, 200, 3] > 200
    assert arr[200, 200, 0] > 180  # red channel preserved

def test_uv_projection_aligns_to_character_bounds(tmp_path):
    client = Hunyuan3DClient()

    # 1. Create character mesh (height 1.8m, width 0.6m)
    box = trimesh.creation.box(extents=(0.6, 0.3, 1.8))
    raw_glb = str(tmp_path / "char.glb")
    box.export(raw_glb)

    # 2. Image with padded margins (character occupies only middle 50%)
    img = Image.new("RGBA", (500, 500), (255, 255, 255, 0)) # transparent background
    draw = ImageDraw.Draw(img)
    # Head (top of character)
    draw.ellipse([200, 80, 300, 150], fill=(240, 190, 150, 255))
    # Red Shirt (chest/torso)
    draw.rectangle([180, 150, 320, 300], fill=(220, 30, 30, 255))
    # Legs
    draw.rectangle([190, 300, 240, 420], fill=(40, 40, 120, 255))
    draw.rectangle([260, 300, 310, 420], fill=(40, 40, 120, 255))

    img_path = str(tmp_path / "padded_char.png")
    img.save(img_path)

    out_glb = str(tmp_path / "projected.glb")
    client.project_texture_to_mesh(
        glb_path=raw_glb,
        image_path=img_path,
        output_glb_path=out_glb
    )

    assert Path(out_glb).exists()
    scene = trimesh.load(out_glb)
    mesh = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene

    # Verify UV coordinates exist and are non-trivial
    assert hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
    uvs = mesh.visual.uv

    # Head vertices (top of 3D mesh, height > 1.5) should map to upper UVs where head is located
    # not squashed into the middle
    verts = mesh.vertices
    head_mask = verts[:, 1] > 1.5
    assert np.any(head_mask)
    head_uv_v = uvs[head_mask, 1]
    # In glTF 2.0 specification, V=0.0 is top (head), V=1.0 is bottom (feet)
    assert np.mean(head_uv_v) < 0.35
