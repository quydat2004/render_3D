import pytest
from pathlib import Path
import trimesh
from services.inference.multi_layer_character import (
    generate_bony_base_body,
    generate_clothing_shell,
    generate_hair_cards,
    MultiLayerCharacter
)
from services.inference.pbr_texture_engine import PBRTextureEngine
from services.inference.procedural_generator import Procedural3DGenerator

def test_multi_layer_character_generation(tmp_path):
    gen = Procedural3DGenerator()
    # Test high-level character generator producing multi-layered glb
    glb_out = tmp_path / "test_aaa_char.glb"

    res_scene = gen.generate_multi_layer_character(
        prompt="Ma nữ kinh dị áo dài rách tóc đen dài",
        output_glb_path=str(glb_out)
    )

    assert glb_out.exists()
    assert glb_out.stat().st_size > 10000

    # Load back with trimesh to inspect scenes and geometries
    scene = trimesh.load(str(glb_out))
    assert isinstance(scene, trimesh.Scene)
    # Must have multiple geometry primitives: body, clothing, hair
    assert len(scene.geometry) >= 2
