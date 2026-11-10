import os
from pathlib import Path
import pytest
import trimesh
from services.dcc.blender_pipeline_processor import BlenderPipelineProcessor

def test_generate_blender_script_contains_key_operations(tmp_path):
    processor = BlenderPipelineProcessor(blender_path="blender")
    script = processor.generate_blender_script(
        input_path="input.obj",
        output_path="output.glb",
        target_faces=25000,
        turntable_prefix=str(tmp_path / "turntable")
    )
    assert "bpy.ops.object.quadriflow_remesh" in script
    assert "target_faces=25000" in script
    assert "bpy.ops.uv.smart_project" in script
    assert "bpy.ops.export_scene.gltf" in script
    assert "TurntableCam" in script

def test_fallback_geometry_processor_with_trimesh(tmp_path):
    # Test fallback processor handles 3D mesh when blender is not in PATH
    processor = BlenderPipelineProcessor(blender_path=None)

    # Create a dense sample sphere (2560 faces)
    sample_mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    input_obj = tmp_path / "test_sphere.obj"
    sample_mesh.export(str(input_obj))

    output_glb = tmp_path / "test_sphere_out.glb"
    turntable_dir = tmp_path / "turntable_renders"
    turntable_dir.mkdir(exist_ok=True)

    result = processor.process_mesh(
        input_path=str(input_obj),
        output_path=str(output_glb),
        target_faces=500,
        turntable_dir=str(turntable_dir)
    )

    assert result["success"] is True
    assert Path(output_glb).exists()
    assert Path(output_glb).stat().st_size > 0
    assert result["output_faces"] <= 550
    assert len(result["turntable_frames"]) == 8
    for frame in result["turntable_frames"]:
        assert Path(frame).exists()
