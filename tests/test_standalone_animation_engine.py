import pytest
from pathlib import Path
import trimesh
from services.dcc.character_rigging_engine import CharacterRiggingEngine

def test_rig_and_animate_character_creates_glb(tmp_path):
    engine = CharacterRiggingEngine()

    # Create humanoid-like test mesh (cylinder torso + head)
    box = trimesh.creation.box(extents=(0.6, 0.3, 1.75))
    input_glb = str(tmp_path / "test_char.glb")
    box.export(input_glb)

    output_glb = str(tmp_path / "test_char_animated.glb")
    res_path = engine.rig_and_animate_character(
        input_glb=input_glb,
        output_glb=output_glb,
        animations=["walk", "idle", "run", "attack"]
    )

    assert Path(res_path).exists()
    assert Path(res_path).stat().st_size > 1000

    # Read glb structure
    import json
    import struct
    data = Path(res_path).read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67  # 'glTF'
    assert version == 2

    chunk_len, chunk_type = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20+chunk_len].decode("utf-8"))

    # Verify skin and animations exist
    assert "skins" in gltf
    assert len(gltf["skins"]) > 0
    assert "animations" in gltf
    assert len(gltf["animations"]) >= 4

    anim_names = [a.get("name", "").lower() for a in gltf["animations"]]
    assert "walk" in anim_names
    assert "idle" in anim_names
    assert "run" in anim_names
    assert "attack" in anim_names
