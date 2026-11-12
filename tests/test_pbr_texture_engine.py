import pytest
import numpy as np
from PIL import Image
from services.inference.pbr_texture_engine import PBRTextureEngine, PBRTextureSet

def test_height_to_normal_map():
    engine = PBRTextureEngine()
    # Create synthetic height array with a central bump
    y, x = np.ogrid[:128, :128]
    height = np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 20.0**2)).astype(np.float32)

    normal_img = engine.height_to_normal_map(height, strength=2.0)
    assert isinstance(normal_img, Image.Image)
    assert normal_img.size == (128, 128)
    assert normal_img.mode == "RGB"

    normal_arr = np.array(normal_img)
    # Blue channel (Z direction) should always be positive (> 128) in tangent-space normal maps
    assert np.all(normal_arr[:, :, 2] >= 120)

def test_hair_alpha_texture():
    engine = PBRTextureEngine()
    hair_img = engine.generate_hair_alpha_texture(width=128, height=256)
    assert isinstance(hair_img, Image.Image)
    assert hair_img.size == (128, 256)
    assert hair_img.mode == "RGBA"

    arr = np.array(hair_img)
    alpha = arr[:, :, 3]
    # Must have both transparent areas (0) and hair strand areas (255)
    assert np.any(alpha == 0)
    assert np.any(alpha > 200)

def test_torn_cloth_alpha():
    engine = PBRTextureEngine()
    cloth_img = engine.generate_torn_cloth_alpha(width=128, height=128)
    assert isinstance(cloth_img, Image.Image)
    assert cloth_img.size == (128, 128)
    assert cloth_img.mode == "RGBA"

    arr = np.array(cloth_img)
    alpha = arr[:, :, 3]
    # Top should be solid fabric (255), bottom hem should have cuts (0)
    assert np.mean(alpha[:40, :]) > 200
    assert np.any(alpha[-20:, :] == 0)

def test_generate_character_textures_set():
    engine = PBRTextureEngine()
    textures = engine.generate_character_textures("ghost_robed", width=256, height=256)
    assert isinstance(textures, PBRTextureSet)
    assert textures.albedo.size == (256, 256)
    assert textures.normal.size == (256, 256)
    assert textures.roughness.size == (256, 256)
    assert textures.alpha_hair.size == (256, 256)
    assert textures.alpha_cloth.size == (256, 256)
