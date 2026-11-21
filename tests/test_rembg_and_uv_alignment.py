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

def test_remove_background_cleans_attached_floor_shadow(tmp_path):
    client = Hunyuan3DClient()
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Character body
    draw.rectangle([150, 100, 250, 320], fill=(220, 20, 20))
    # Attached dark floor contact shadow under/behind feet
    draw.ellipse([140, 318, 260, 345], fill=(70, 70, 70))

    clean_img = client.remove_background(img)
    arr = np.array(clean_img)
    # Character body preserved
    assert arr[200, 200, 3] > 200
    # Floor contact shadow cleared to transparent
    assert arr[335, 200, 3] == 0

def test_paint_pipeline_is_the_only_texture_path():
    """Texture comes from Hunyuan3D-Paint. There is no planar photo projection anywhere."""
    client = Hunyuan3DClient()
    assert not hasattr(client, "project_texture_to_mesh")
    assert Hunyuan3DClient.PAINT_SPACE_ID
    src = Path(Hunyuan3DClient.texture_mesh_with_paint.__code__.co_filename)
    text = src.read_text(encoding="utf-8")
    assert "project_texture_to_mesh" not in text
    assert "/texture_mesh" in text
