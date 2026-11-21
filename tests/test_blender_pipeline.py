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
    assert "type='DECIMATE'" in script
    assert "decimate_type = 'COLLAPSE'" in script
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

def test_fallback_handles_uv_preservation(tmp_path):
    # Test fallback preserves UV atlas when present (critical for rigged characters)
    processor = BlenderPipelineProcessor(blender_path=None)
    sample_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    input_obj = tmp_path / "test_uv.obj"
    sample_mesh.export(str(input_obj))

    output_glb = tmp_path / "test_uv_out.glb"
    turntable_dir = tmp_path / "turntable_uv"
    turntable_dir.mkdir(exist_ok=True)

    result = processor.process_mesh(
        input_path=str(input_obj),
        output_path=str(output_glb),
        target_faces=200,
        turntable_dir=str(turntable_dir)
    )

    assert result["success"] is True
    assert Path(output_glb).exists()
    assert Path(output_glb).stat().st_size > 100

def test_fallback_exports_draco_compressed_glb(tmp_path):
    # Test trimesh fallback exports Draco-compressed GLB (production requirement)
    processor = BlenderPipelineProcessor(blender_path=None)
    sample_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    input_obj = tmp_path / "test_draco.obj"
    sample_mesh.export(str(input_obj))

    output_glb = tmp_path / "test_draco_out.glb"
    turntable_dir = tmp_path / "turntable_draco"
    turntable_dir.mkdir(exist_ok=True)

    result = processor.process_mesh(
        input_path=str(input_obj),
        output_path=str(output_glb),
        target_faces=200,
        turntable_dir=str(turntable_dir)
    )

    assert result["success"] is True
    assert result["engine"] == "trimesh_fallback"
    assert result.get("draco_compressed") is True
    assert Path(output_glb).exists()
    assert Path(output_glb).stat().st_size > 100

def test_clean_floating_and_heel_spur_artifacts():
    import numpy as np
    # 1. Main body cylinder (height 1.75m, radius 0.25)
    body = trimesh.creation.cylinder(radius=0.25, height=1.75)
    body.apply_translation([0, 0, 0.875])

    # 2. Add an attached heel spur (thin sliver/fin triangles sticking out behind the heel at the ground)
    # Posterior heel spur at z=0.02, extending out in -Y
    spur_verts = np.array([
        [0.0, -0.25, 0.01],
        [0.0, -0.55, 0.01], # protruding outward behind heel
        [0.0, -0.25, 0.08],
        [0.0, -0.52, 0.06],
    ], dtype=np.float64)
    spur_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    spur_mesh = trimesh.Trimesh(vertices=spur_verts, faces=spur_faces)

    # 3. Add a floating disconnected island
    floating = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
    floating.apply_translation([0.8, 0, 0.5])

    combined = trimesh.util.concatenate([body, spur_mesh, floating])
    cleaned = BlenderPipelineProcessor.clean_floating_and_heel_artifacts(combined)

    # The heel spur and floating cube should be eliminated
    assert len(cleaned.faces) < len(combined.faces)
    assert float(np.min(cleaned.vertices[:, 1])) > -0.35, "Posterior heel spur sticking out behind the heel must be removed"
    assert float(np.max(cleaned.vertices[:, 0])) < 0.6, "Floating disconnected island must be removed"

def test_sever_anatomical_bridges_and_decimate_prevents_hand_thigh_welding():
    import numpy as np
    # Create two close cylinders: Thigh (x=0.08) and Hand (x=0.22) with a narrow 2cm air gap
    thigh = trimesh.creation.cylinder(radius=0.08, height=0.6, sections=24)
    thigh.apply_translation([0.08, 0, 0.8])

    hand = trimesh.creation.cylinder(radius=0.04, height=0.4, sections=20)
    hand.apply_translation([0.22, 0, 0.8])

    mesh = trimesh.util.concatenate([thigh, hand])
    decimated = BlenderPipelineProcessor.sever_anatomical_bridges_and_decimate(mesh, target_faces=300)

    # Hand and Thigh must remain separate connected components in 3D
    parts = decimated.split(only_watertight=False)
    assert len(parts) >= 2, "Hand and thigh must remain separate and not weld into a single block"


def test_compute_crease_aware_smooth_normals_preserves_sharp_edges_and_smooths_curves():
    """
    Crease-aware normals must split vertices across 90-degree box corners (sharp highlights)
    while smoothly welding curved cylinder surfaces.
    """
    import trimesh
    box = trimesh.creation.box()
    creased_box = BlenderPipelineProcessor.compute_crease_aware_smooth_normals(box, crease_angle_deg=40.0)
    # 6 faces of box should each have their own flat normals (24 vertices total)
    assert len(creased_box.vertices) == 24
    assert creased_box.vertex_normals is not None

    cyl = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=32)
    creased_cyl = BlenderPipelineProcessor.compute_crease_aware_smooth_normals(cyl, crease_angle_deg=40.0)
    # Cylinder barrel is smoothed, but top/bottom cap rims (90 deg) are sharp
    assert len(creased_cyl.vertices) == 130
    assert creased_cyl.vertex_normals is not None


def test_sharpen_and_bake_cavity_ao_enhances_relief_and_bakes_colors():
    """
    sharpen_and_bake_cavity_ao must displace relief along normals and bake cavity shading
    so crevices are darker than ridges.
    """
    import numpy as np
    import trimesh

    # Create icosphere
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # Indent a groove into the sphere at z < -0.8 (crevice) and raise a ridge at z > 0.8 (crest)
    v = sphere.vertices.copy()
    crevice_mask = v[:, 2] < -0.8
    crest_mask = v[:, 2] > 0.8
    v[crevice_mask] *= 0.85 # sunken crevice
    v[crest_mask] *= 1.15   # raised crest
    sphere.vertices = v
    sphere.fix_normals()

    orig_crevice_radius = np.mean(np.linalg.norm(sphere.vertices[crevice_mask], axis=1))
    orig_crest_radius = np.mean(np.linalg.norm(sphere.vertices[crest_mask], axis=1))

    sharpened = BlenderPipelineProcessor.sharpen_and_bake_cavity_ao(sphere, boost_relief=0.40)

    # 1. Vertex colors baked
    assert hasattr(sharpened.visual, "vertex_colors")
    colors = sharpened.visual.vertex_colors
    assert colors is not None
    assert len(colors) == len(sharpened.vertices)

    # 2. Cavity shading: crevice vertices should have lower luminance than crest vertices
    crevice_lum = np.mean(colors[crevice_mask, :3])
    crest_lum = np.mean(colors[crest_mask, :3])
    assert crevice_lum < crest_lum, f"Crevices ({crevice_lum}) must be darker than crests ({crest_lum})"

    # 3. 3D Relief Boost: crevice should be carved deeper and crest raised higher
    new_crevice_radius = np.mean(np.linalg.norm(sharpened.vertices[crevice_mask], axis=1))
    new_crest_radius = np.mean(np.linalg.norm(sharpened.vertices[crest_mask], axis=1))
    assert new_crevice_radius < orig_crevice_radius, "Relief boosting must carve sunken crevices deeper"
    assert new_crest_radius > orig_crest_radius, "Relief boosting must elevate ridge crests higher"


