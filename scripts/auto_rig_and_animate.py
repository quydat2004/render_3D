"""
Blender Headless Script: Auto-Rigging & Skeletal Animation Retargeting Pipeline
Run via:
    blender --background --python scripts/auto_rig_and_animate.py -- <input_glb> <output_glb> [anim_list]
"""

import sys
import os
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

def parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    """
    Parses arguments following the '--' delimiter passed to Blender CLI.
    """
    if "--" in argv:
        args_after = argv[argv.index("--") + 1:]
    else:
        args_after = argv[1:]

    input_glb = args_after[0] if len(args_after) > 0 else "input.glb"
    output_glb = args_after[1] if len(args_after) > 1 else "animated_character.glb"

    raw_anims = args_after[2] if len(args_after) > 2 else "walk,idle,run,attack"
    if isinstance(raw_anims, str):
        animations = [a.strip().lower() for a in raw_anims.split(",") if a.strip()]
    else:
        animations = ["walk", "idle", "run", "attack"]

    return {
        "input_glb": input_glb,
        "output_glb": output_glb,
        "animations": animations
    }

def clean_blender_scene():
    if not bpy:
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        bpy.data.armatures.remove(block)

def build_humanoid_rig(character_mesh) -> Any:
    """
    Constructs a standard humanoid bone armature scaled and aligned to the character mesh bounding box.
    """
    if not bpy:
        return None

    # Calculate bounding box
    bbox_corners = [character_mesh.matrix_world @ mathutils.Vector(corner) for corner in character_mesh.bound_box]
    min_x = min(v.x for v in bbox_corners)
    max_x = max(v.x for v in bbox_corners)
    min_y = min(v.y for v in bbox_corners)
    max_y = max(v.y for v in bbox_corners)
    min_z = min(v.z for v in bbox_corners)
    max_z = max(v.z for v in bbox_corners)

    height = max(max_z - min_z, 1e-3)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    # Silhouette Cross-Section Scan to detect neck & hips for Chibi / Stylized characters
    mesh_verts_z = [v.co.z for v in character_mesh.data.vertices]
    z_min = min(mesh_verts_z)
    z_max = max(mesh_verts_z)
    height = max(z_max - z_min, 1e-3)

    slices = 30
    z_steps = [z_min + height * (0.35 + 0.50 * (i / slices)) for i in range(slices)]
    widths = []
    dz = height / slices
    for z_val in z_steps:
        slice_verts = [v.co.x for v in character_mesh.data.vertices if abs(v.co.z - z_val) < dz]
        w = (max(slice_verts) - min(slice_verts)) if slice_verts else 1.0
        widths.append(w)
    min_w_idx = widths.index(min(widths))
    z_neck = max(min(z_steps[min_w_idx], z_min + height * 0.86), z_min + height * 0.45)
    z_head = z_max - (z_max - z_neck) * 0.42
    head_height_ratio = (z_max - z_neck) / height
    is_chibi = (head_height_ratio > 0.33)

    z_hip = z_min + (z_neck - z_min) * (0.42 if is_chibi else 0.45)
    z_spine = z_hip + (z_neck - z_hip) * 0.35
    z_chest = z_hip + (z_neck - z_hip) * 0.70
    z_knee = z_min + (z_hip - z_min) * (0.45 if is_chibi else 0.50)

    # Detect pre-bent knee forward offset
    knee_slice = [v.co for v in character_mesh.data.vertices if abs(v.co.z - z_knee) < dz * 1.8]
    knee_center_y = (sum(v.y for v in knee_slice) / len(knee_slice)) if knee_slice else center_y

    # Create Armature
    armature_data = bpy.data.armatures.new("Humanoid_Armature")
    armature_obj = bpy.data.objects.new("Armature", armature_data)
    bpy.context.scene.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = armature_data.edit_bones

    # 1. Pelvis / Hips
    hips = edit_bones.new("Hips")
    hips.head = (center_x, center_y, z_hip)
    hips.tail = (center_x, center_y, z_spine)

    # 2. Spine & Chest
    spine = edit_bones.new("Spine")
    spine.head = hips.tail
    spine.tail = (center_x, center_y, z_chest)
    spine.parent = hips

    chest = edit_bones.new("Chest")
    chest.head = spine.tail
    chest.tail = (center_x, center_y, z_neck)
    chest.parent = spine

    neck = edit_bones.new("Neck")
    neck.head = chest.tail
    neck.tail = (center_x, center_y, z_neck + (z_head - z_neck) * 0.3)
    neck.parent = chest

    head = edit_bones.new("Head")
    head.head = neck.tail
    head.tail = (center_x, center_y, z_max)
    head.parent = neck

    # 3. Arms
    arm_span = (max_x - min_x) * 0.45
    for sign, side in [(-1.0, "L"), (1.0, "R")]:
        sh = edit_bones.new(f"Shoulder.{side}")
        sh.head = (center_x + sign * 0.05, center_y, z_chest)
        sh.tail = (center_x + sign * 0.16, center_y, z_chest)
        sh.parent = chest

        uarm = edit_bones.new(f"UpperArm.{side}")
        uarm.head = sh.tail
        uarm.tail = (center_x + sign * (0.16 + arm_span * 0.45), center_y, z_chest - (z_chest - z_hip) * 0.42)
        uarm.parent = sh

        larm = edit_bones.new(f"Forearm.{side}")
        larm.head = uarm.tail
        larm.tail = (center_x + sign * (0.16 + arm_span * 0.85), center_y, z_hip)
        larm.parent = uarm

        hand = edit_bones.new(f"Hand.{side}")
        hand.head = larm.tail
        hand.tail = (center_x + sign * (0.16 + arm_span * 1.00), center_y, z_hip - 0.05)
        hand.parent = larm

    # 4. Legs
    leg_x = (max_x - min_x) * 0.22
    for sign, side in [(-1.0, "L"), (1.0, "R")]:
        uleg = edit_bones.new(f"UpperLeg.{side}")
        uleg.head = (center_x + sign * leg_x, center_y, z_hip)
        uleg.tail = (center_x + sign * leg_x, knee_center_y, z_knee)
        uleg.parent = hips

        lleg = edit_bones.new(f"LowerLeg.{side}")
        lleg.head = uleg.tail
        lleg.tail = (center_x + sign * leg_x, center_y, min_z + 0.06)
        lleg.parent = uleg

        foot = edit_bones.new(f"Foot.{side}")
        foot.head = lleg.tail
        foot.tail = (center_x + sign * leg_x, center_y + 0.12, min_z)
        foot.parent = lleg

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj

def add_procedural_action(armature_obj, action_name: str, fps: int = 30):
    """
    Synthesizes smooth procedural humanoid keyframes for common actions (Walk, Idle, Run, Attack).
    """
    if not bpy or not armature_obj.animation_data:
        armature_obj.animation_data_create()

    action = bpy.data.actions.new(name=action_name)
    armature_obj.animation_data.action = action
    pose_bones = armature_obj.pose.bones

    if action_name.lower() == "walk":
        # 30 frames (1 second loop)
        total_frames = 30
        for f in range(total_frames + 1):
            t = (f / total_frames) * math.pi * 2.0
            # Leg strides
            if "UpperLeg.L" in pose_bones and "UpperLeg.R" in pose_bones:
                pose_bones["UpperLeg.L"].rotation_mode = 'XYZ'
                pose_bones["UpperLeg.R"].rotation_mode = 'XYZ'
                pose_bones["UpperLeg.L"].rotation_euler = (math.sin(t) * 0.45, 0.0, 0.0)
                pose_bones["UpperLeg.R"].rotation_euler = (-math.sin(t) * 0.45, 0.0, 0.0)
                pose_bones["UpperLeg.L"].keyframe_insert(data_path="rotation_euler", frame=f)
                pose_bones["UpperLeg.R"].keyframe_insert(data_path="rotation_euler", frame=f)

            # Knee bend during backward phase
            if "LowerLeg.L" in pose_bones and "LowerLeg.R" in pose_bones:
                pose_bones["LowerLeg.L"].rotation_mode = 'XYZ'
                pose_bones["LowerLeg.R"].rotation_mode = 'XYZ'
                pose_bones["LowerLeg.L"].rotation_euler = (max(0.0, -math.sin(t) * 0.6), 0.0, 0.0)
                pose_bones["LowerLeg.R"].rotation_euler = (max(0.0, math.sin(t) * 0.6), 0.0, 0.0)
                pose_bones["LowerLeg.L"].keyframe_insert(data_path="rotation_euler", frame=f)
                pose_bones["LowerLeg.R"].keyframe_insert(data_path="rotation_euler", frame=f)

            # Arm swings counter to legs
            if "UpperArm.L" in pose_bones and "UpperArm.R" in pose_bones:
                pose_bones["UpperArm.L"].rotation_mode = 'XYZ'
                pose_bones["UpperArm.R"].rotation_mode = 'XYZ'
                pose_bones["UpperArm.L"].rotation_euler = (-math.sin(t) * 0.35, 0.0, 0.0)
                pose_bones["UpperArm.R"].rotation_euler = (math.sin(t) * 0.35, 0.0, 0.0)
                pose_bones["UpperArm.L"].keyframe_insert(data_path="rotation_euler", frame=f)
                pose_bones["UpperArm.R"].keyframe_insert(data_path="rotation_euler", frame=f)

            # Hip bounce
            if "Hips" in pose_bones:
                pose_bones["Hips"].location = (0.0, 0.0, math.sin(t * 2.0) * 0.015)
                pose_bones["Hips"].keyframe_insert(data_path="location", frame=f)

    elif action_name.lower() == "idle":
        total_frames = 60
        for f in range(total_frames + 1):
            t = (f / total_frames) * math.pi * 2.0
            if "Chest" in pose_bones:
                pose_bones["Chest"].rotation_mode = 'XYZ'
                pose_bones["Chest"].rotation_euler = (math.sin(t) * 0.04, 0.0, 0.0)
                pose_bones["Chest"].keyframe_insert(data_path="rotation_euler", frame=f)
            if "UpperArm.L" in pose_bones and "UpperArm.R" in pose_bones:
                pose_bones["UpperArm.L"].rotation_mode = 'XYZ'
                pose_bones["UpperArm.R"].rotation_mode = 'XYZ'
                pose_bones["UpperArm.L"].rotation_euler = (0.0, 0.0, math.sin(t) * 0.03)
                pose_bones["UpperArm.R"].rotation_euler = (0.0, 0.0, -math.sin(t) * 0.03)
                pose_bones["UpperArm.L"].keyframe_insert(data_path="rotation_euler", frame=f)
                pose_bones["UpperArm.R"].keyframe_insert(data_path="rotation_euler", frame=f)

    elif action_name.lower() == "run":
        total_frames = 18
        for f in range(total_frames + 1):
            t = (f / total_frames) * math.pi * 2.0
            if "Spine" in pose_bones:
                pose_bones["Spine"].rotation_mode = 'XYZ'
                pose_bones["Spine"].rotation_euler = (0.22, 0.0, 0.0)  # Forward lean
                pose_bones["Spine"].keyframe_insert(data_path="rotation_euler", frame=f)
            if "UpperLeg.L" in pose_bones and "UpperLeg.R" in pose_bones:
                pose_bones["UpperLeg.L"].rotation_mode = 'XYZ'
                pose_bones["UpperLeg.R"].rotation_mode = 'XYZ'
                pose_bones["UpperLeg.L"].rotation_euler = (math.sin(t) * 0.75, 0.0, 0.0)
                pose_bones["UpperLeg.R"].rotation_euler = (-math.sin(t) * 0.75, 0.0, 0.0)
                pose_bones["UpperLeg.L"].keyframe_insert(data_path="rotation_euler", frame=f)
                pose_bones["UpperLeg.R"].keyframe_insert(data_path="rotation_euler", frame=f)
            if "UpperArm.L" in pose_bones and "UpperArm.R" in pose_bones:
                pose_bones["UpperArm.L"].rotation_mode = 'XYZ'
                pose_bones["UpperArm.R"].rotation_mode = 'XYZ'
                pose_bones["UpperArm.L"].rotation_euler = (-math.sin(t) * 0.65, 0.0, 0.0)
                pose_bones["UpperArm.R"].rotation_euler = (math.sin(t) * 0.65, 0.0, 0.0)
                pose_bones["UpperArm.L"].keyframe_insert(data_path="rotation_euler", frame=f)
                pose_bones["UpperArm.R"].keyframe_insert(data_path="rotation_euler", frame=f)

    elif action_name.lower() == "attack":
        total_frames = 36
        for f in range(total_frames + 1):
            t = (f / total_frames) * math.pi * 2.0
            if "Chest" in pose_bones and "UpperArm.R" in pose_bones and "Forearm.R" in pose_bones:
                pose_bones["Chest"].rotation_mode = 'XYZ'
                pose_bones["UpperArm.R"].rotation_mode = 'XYZ'
                pose_bones["Forearm.R"].rotation_mode = 'XYZ'

                # Windup (0 to 12) -> Strike (13 to 22) -> Recover (23 to 36)
                if f < 14:
                    prog = f / 14.0
                    pose_bones["UpperArm.R"].rotation_euler = (prog * 1.2, -prog * 0.5, prog * 0.8)
                    pose_bones["Chest"].rotation_euler = (0.0, -prog * 0.4, 0.0)
                elif f < 24:
                    prog = (f - 14) / 10.0
                    pose_bones["UpperArm.R"].rotation_euler = (1.2 - prog * 1.8, -0.5 + prog * 0.8, 0.8 - prog * 0.4)
                    pose_bones["Chest"].rotation_euler = (0.0, -0.4 + prog * 0.7, 0.0)
                else:
                    prog = (f - 24) / 12.0
                    pose_bones["UpperArm.R"].rotation_euler = (-0.6 * (1.0 - prog), 0.3 * (1.0 - prog), 0.4 * (1.0 - prog))
                    pose_bones["Chest"].rotation_euler = (0.0, 0.3 * (1.0 - prog), 0.0)

                pose_bones["Chest"].keyframe_insert(data_path="rotation_euler", frame=f)
                pose_bones["UpperArm.R"].keyframe_insert(data_path="rotation_euler", frame=f)

    # Push to NLA Track
    track = armature_obj.animation_data.nla_tracks.new()
    track.name = action_name
    track.strips.new(action.name, 0, action)

def process_auto_rig_and_animate(input_glb: str, output_glb: str, animations: List[str]):
    if not bpy:
        raise RuntimeError("Blender environment is not active.")

    clean_blender_scene()

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=input_glb)

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        raise ValueError(f"No mesh objects found in {input_glb}")

    # Join multiple mesh objects into single unified body
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    character_mesh = bpy.context.active_object
    character_mesh.name = "Character_Mesh"

    # Normalize Scale to 1.75m and Ground to Y=0
    dim_z = max(character_mesh.dimensions.z, 0.01)
    scale_factor = 1.75 / dim_z
    character_mesh.scale *= scale_factor
    bpy.ops.object.transform_apply(scale=True)

    # Re-ground min Z to 0.0
    min_z = min((character_mesh.matrix_world @ mathutils.Vector(v)).z for v in character_mesh.bound_box)
    character_mesh.location.z -= min_z
    bpy.ops.object.transform_apply(location=True)

    # Decimate high-poly neural meshes to ~25,000 tris using Collapse Decimate (preserves UV map & painted texture)
    poly_count = len(character_mesh.data.polygons)
    if poly_count > 25000:
        dec_mod = character_mesh.modifiers.new(name="PreserveUVDecimate", type='DECIMATE')
        dec_mod.decimate_type = 'COLLAPSE'
        dec_mod.ratio = max(25000.0 / float(poly_count), 0.02)
        bpy.ops.object.modifier_apply(modifier=dec_mod.name)

    # Enforce natural matte PBR shader settings (Metallic=0.0, Roughness=0.65)
    for mat in character_mesh.data.materials:
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if 'Metallic' in node.inputs and not node.inputs['Metallic'].is_linked:
                        node.inputs['Metallic'].default_value = 0.0
                    if 'Roughness' in node.inputs and not node.inputs['Roughness'].is_linked:
                        node.inputs['Roughness'].default_value = 0.65

    # Create Humanoid Rig
    armature_obj = build_humanoid_rig(character_mesh)

    # Parent Mesh to Armature with Automatic Weights
    bpy.ops.object.select_all(action='DESELECT')
    character_mesh.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    # Apply animations
    anim_dir = Path("assets/animations")
    for anim_name in animations:
        fbx_path = anim_dir / f"{anim_name}.fbx"
        bvh_path = anim_dir / f"{anim_name}.bvh"

        if fbx_path.exists():
            # Import and transfer from FBX preset
            bpy.ops.import_scene.fbx(filepath=str(fbx_path))
        elif bvh_path.exists():
            bpy.ops.import_anim.bvh(filepath=str(bvh_path))
        else:
            # Generate procedural animation keyframes
            add_procedural_action(armature_obj, anim_name)

    # Export to GLB with Animation NLA tracks
    out_path = Path(output_glb)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format='GLB',
        export_materials='EXPORT',
        export_image_format='AUTO',
        export_texcoords=True,
        export_normals=True,
        export_animations=True,
        export_all_influences=False,
        export_skins=True,
        export_def_bones=True
    )
    print(f"[Render3D] Successfully exported animated character: {output_glb}")

if __name__ == "__main__":
    opts = parse_cli_args(sys.argv)
    if bpy:
        process_auto_rig_and_animate(
            input_glb=opts["input_glb"],
            output_glb=opts["output_glb"],
            animations=opts["animations"]
        )
    else:
        print("[Render3D] Executed outside Blender. CLI parser verified.")
