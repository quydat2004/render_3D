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

def test_enhance_micro_details_teeth_and_collar():
    engine = PBRTextureEngine()
    # Create synthetic face/collar texture: lip color bleeding into teeth, and neck-to-collar border
    img_arr = np.zeros((128, 128, 3), dtype=np.uint8)
    # Neck skin (top)
    img_arr[:40, :] = [230, 185, 160]
    # Red bleeding into dental zone at y=30..35
    img_arr[30:36, 50:78] = [210, 100, 110] # red-peach bleed on teeth
    # Collar garment (bottom)
    img_arr[40:, :] = [35, 45, 60]

    input_img = Image.fromarray(img_arr, mode="RGB")
    enhanced_img = engine.enhance_micro_details(input_img, sharpen_strength=1.45)
    enhanced_arr = np.array(enhanced_img)

    # Dental zone should have reduced chroma / anti-bleed clamping
    orig_chroma = np.max(img_arr[32, 64]) - np.min(img_arr[32, 64])
    enh_chroma = int(np.max(enhanced_arr[32, 64])) - int(np.min(enhanced_arr[32, 64]))
    assert enh_chroma < orig_chroma, "Dental enamel anti-bleed should reduce color fringing"

    # Collar transition at y=39 vs y=41 should maintain strong contrast without muddy blur
    diff_neck_collar = np.linalg.norm(enhanced_arr[38, 64].astype(float) - enhanced_arr[42, 64].astype(float))
    assert diff_neck_collar > 100.0, "Collar trim edge should have crisp contrast"

def test_repair_occluded_regions_chin_armpit_heel():
    engine = PBRTextureEngine()
    # Create image with washed out chalky white blotches in chin (y~0.25) and heel (y~0.90)
    arr = np.full((128, 128, 3), [190, 80, 50], dtype=np.uint8) # warm tone
    # Washed out chalky white in chin zone (y in [25, 38])
    arr[25:38, 45:80] = [230, 228, 226]
    # Washed out gray in heel zone (y in [105, 120])
    arr[105:120, 30:70] = [215, 215, 215]

    repaired = engine.repair_occluded_regions(Image.fromarray(arr, mode="RGB"))
    rep_arr = np.array(repaired)

    # Inpainted pixels should no longer be chalky neutral (chroma > 10, shaded with cavity AO)
    chin_chroma = int(np.max(rep_arr[30, 60])) - int(np.min(rep_arr[30, 60]))
    heel_chroma = int(np.max(rep_arr[110, 50])) - int(np.min(rep_arr[110, 50]))
    assert chin_chroma > 15, "Chin crevice should be inpainted from neighboring skin/body color"
    assert heel_chroma > 15, "Heel/sole crevice should be inpainted from neighboring shoe color"

def test_synthesize_material_zone_maps_distinguishes_fabrics_and_skin():
    engine = PBRTextureEngine()
    # Synthesize image with distinct skin, sportswear fabric, hair, shoes, and wet eyes
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    # Hair (y < 20)
    arr[:20, :] = [25, 20, 20]
    # Eyes/teeth (y in [20, 30], x in [55, 75])
    arr[20:30, 55:75] = [245, 245, 245]
    # Skin (y in [30, 50])
    arr[30:50, :] = [225, 180, 155]
    # Sportswear fabric (y in [50, 105])
    arr[50:105, :] = [40, 50, 65]
    # Shoes (y > 110)
    arr[110:, :] = [200, 40, 40]

    rough_img, spec_img, ao_img, mr_img = engine.synthesize_material_zone_maps(Image.fromarray(arr, mode="RGB"))
    rough_arr = np.array(rough_img)

    # Verify material zones exhibit distinct roughness
    rough_eyes = float(np.mean(rough_arr[22:28, 58:72]))
    rough_skin = float(np.mean(rough_arr[35:45, :]))
    rough_fabric = float(np.mean(rough_arr[65:85, :]))
    rough_shoes = float(np.mean(rough_arr[115:, :]))

    assert rough_eyes < 50.0, f"Eyes/teeth must be wet high-gloss (got {rough_eyes})"
    assert 110.0 <= rough_skin <= 165.0, f"Skin must be matte-satin (got {rough_skin})"
    assert rough_fabric > 185.0, f"Sportswear fabric must be high roughness (got {rough_fabric})"
    assert rough_shoes < 120.0, f"Shoes/soles must be semi-gloss rubber/leather (got {rough_shoes})"
    assert rough_fabric > rough_skin, "Sportswear fabric must be rougher than skin"
