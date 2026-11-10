import pytest
import numpy as np
import trimesh
from services.dcc.map_modular_assembler import (
    ModularMapAssembler,
    CoverType,
    CoverHint
)
from services.orchestrator.gemini_orchestrator import LevelLayoutSchema

def test_snap_to_grid():
    assembler = ModularMapAssembler(grid_size_m=2.0)

    raw_pos = [2.03, 3.97, 0.05]
    snapped = assembler.snap_to_grid(raw_pos)
    assert snapped == [2.0, 4.0, 0.0]

    raw_neg = [-1.98, -4.04, 2.01]
    snapped_neg = assembler.snap_to_grid(raw_neg)
    assert snapped_neg == [-2.0, -4.0, 2.0]

def test_classify_cover_elements():
    assembler = ModularMapAssembler()

    # Half-cover crate (height 1.1m)
    crate = trimesh.creation.box(extents=[1.0, 1.0, 1.1])
    hint_half = assembler.classify_cover(crate)
    assert hint_half == CoverType.COVER_HALF

    # Full-cover pillar (height 3.0m)
    pillar = trimesh.creation.box(extents=[0.8, 0.8, 3.0])
    hint_full = assembler.classify_cover(pillar)
    assert hint_full == CoverType.COVER_FULL

    # Very low ground debris (height 0.3m) -> Not cover
    debris = trimesh.creation.box(extents=[0.5, 0.5, 0.3])
    hint_none = assembler.classify_cover(debris)
    assert hint_none == CoverType.NONE

def test_assemble_modular_scene(tmp_path):
    assembler = ModularMapAssembler(grid_size_m=2.0)

    layout = LevelLayoutSchema(
        level_id="test_arena",
        dimensions=[20.0, 20.0],
        grid_size_m=2.0,
        biome="Industrial",
        instances=[
            {"asset_id": "floor_4x4", "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
            {"asset_id": "wall_4x4", "position": [0.0, 0.0, 2.0], "rotation": [0.0, 90.0, 0.0], "scale": [1.0, 1.0, 1.0]}
        ]
    )

    scene_result = assembler.assemble_scene(layout, output_dir=str(tmp_path))
    assert scene_result["instance_count"] == 2
    assert "navmesh" in scene_result
    assert scene_result["navmesh"]["walkable_area_m2"] > 0

def test_generate_map_blender_script():
    assembler = ModularMapAssembler(grid_size_m=2.0)
    layout = LevelLayoutSchema(
        level_id="blender_arena",
        dimensions=[40.0, 40.0],
        grid_size_m=2.0,
        biome="SciFi",
        instances=[
            {"asset_id": "wall_01", "position": [4.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}
        ]
    )

    script = assembler.generate_blender_assembly_script(layout, "output_map.glb")
    assert "bpy.data.objects.new" in script
    assert "linked duplicate" in script.lower() or "mesh_data" in script
    assert "wall_01" in script
    assert "export_scene.gltf" in script
