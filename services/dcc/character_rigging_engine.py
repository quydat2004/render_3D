import os
import json
import struct
import math
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import trimesh

logger = logging.getLogger(__name__)

STANDARD_HUMANOID_BONES = [
    "Root",
    "Pelvis",
    "Spine_01",
    "Spine_02",
    "Neck",
    "Head",
    "Clavicle_L",
    "UpperArm_L",
    "LowerArm_L",
    "Hand_L",
    "Clavicle_R",
    "UpperArm_R",
    "LowerArm_R",
    "Hand_R",
    "Thigh_L",
    "Calf_L",
    "Foot_L",
    "Toe_L",
    "Thigh_R",
    "Calf_R",
    "Foot_R",
    "Toe_R"
]

class HumanoidLandmarks(BaseModel):
    root: List[float] = Field(default=[0.0, 0.0, 0.0])
    pelvis: List[float]
    spine: List[float]
    neck: List[float]
    head: List[float]
    shoulder_l: List[float]
    elbow_l: List[float]
    wrist_l: List[float]
    shoulder_r: List[float]
    elbow_r: List[float]
    wrist_r: List[float]
    hip_l: List[float]
    knee_l: List[float]
    ankle_l: List[float]
    hip_r: List[float]
    knee_r: List[float]
    ankle_r: List[float]

def _axis_angle_to_quat(axis: Tuple[float, float, float], angle_rad: float) -> List[float]:
    ax = np.array(axis, dtype=np.float32)
    norm = np.linalg.norm(ax)
    if norm > 1e-6:
        ax = ax / norm
    s = math.sin(angle_rad * 0.5)
    c = math.cos(angle_rad * 0.5)
    return [float(ax[0] * s), float(ax[1] * s), float(ax[2] * s), float(c)]

class CharacterRiggingEngine:
    def __init__(self):
        pass

    def estimate_landmarks(self, mesh: trimesh.Trimesh) -> HumanoidLandmarks:
        bounds = mesh.bounds
        min_pt = bounds[0]
        max_pt = bounds[1]

        center_x = (min_pt[0] + max_pt[0]) / 2.0
        center_y = (min_pt[1] + max_pt[1]) / 2.0
        height = max_pt[2] - min_pt[2]
        base_z = min_pt[2]

        width = max_pt[0] - min_pt[0]
        shoulder_offset_x = max(width * 0.35, 0.20)
        hip_offset_x = max(width * 0.18, 0.10)

        return HumanoidLandmarks(
            root=[center_x, center_y, base_z],
            pelvis=[center_x, center_y, base_z + height * 0.52],
            spine=[center_x, center_y, base_z + height * 0.68],
            neck=[center_x, center_y, base_z + height * 0.84],
            head=[center_x, center_y, base_z + height * 0.94],
            shoulder_l=[-shoulder_offset_x, center_y, base_z + height * 0.80],
            elbow_l=[-shoulder_offset_x * 1.5, center_y, base_z + height * 0.62],
            wrist_l=[-shoulder_offset_x * 1.9, center_y, base_z + height * 0.45],
            shoulder_r=[shoulder_offset_x, center_y, base_z + height * 0.80],
            elbow_r=[shoulder_offset_x * 1.5, center_y, base_z + height * 0.62],
            wrist_r=[shoulder_offset_x * 1.9, center_y, base_z + height * 0.45],
            hip_l=[-hip_offset_x, center_y, base_z + height * 0.48],
            knee_l=[-hip_offset_x, center_y, base_z + height * 0.26],
            ankle_l=[-hip_offset_x, center_y, base_z + height * 0.05],
            hip_r=[hip_offset_x, center_y, base_z + height * 0.48],
            knee_r=[hip_offset_x, center_y, base_z + height * 0.26],
            ankle_r=[hip_offset_x, center_y, base_z + height * 0.05],
        )

    def clamp_and_normalize_weights(
        self,
        weights: Dict[str, float],
        max_influences: int = 4
    ) -> Dict[str, float]:
        if not weights:
            return {}
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_weights = dict(sorted_weights[:max_influences])
        total = sum(top_weights.values())
        if total <= 0:
            return {k: 1.0 / len(top_weights) for k in top_weights}
        return {k: v / total for k, v in top_weights.items()}

    def generate_rigging_script(
        self,
        mesh_path: str,
        output_glb: str
    ) -> str:
        safe_mesh = mesh_path.replace("\\", "/")
        safe_out = output_glb.replace("\\", "/")

        return f'''import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath="{safe_mesh}")
mesh_obj = [o for o in bpy.context.selected_objects if o.type == 'MESH'][0]

armature_data = bpy.data.armatures.new("Humanoid_Armature")
armature_obj = bpy.data.objects.new("Humanoid_Rig", armature_data)
bpy.context.collection.objects.link(armature_obj)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.mode_set(mode='EDIT')

bones = armature_data.edit_bones
pelvis_bone = bones.new("Pelvis")
pelvis_bone.head = (0, 0, 1.0)
pelvis_bone.tail = (0, 0, 1.2)

head_bone = bones.new("Head")
head_bone.head = (0, 0, 1.6)
head_bone.tail = (0, 0, 1.8)
head_bone.parent = pelvis_bone

bpy.ops.object.mode_set(mode='OBJECT')
mesh_obj.select_set(True)
armature_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.parent_set(type='ARMATURE_AUTO')

def clamp_vertex_weights(obj, max_influences=4):
    for v in obj.data.vertices:
        groups = sorted(v.groups, key=lambda g: g.weight, reverse=True)
        if len(groups) > max_influences:
            for g in groups[max_influences:]:
                obj.vertex_groups[g.group].remove([v.index])

clamp_vertex_weights(mesh_obj)

bpy.ops.export_scene.gltf(
    filepath="{safe_out}",
    export_format='GLB',
    export_skins=True,
    export_all_influences=False
)
'''

    def rig_and_animate_character(
        self,
        input_glb: str,
        output_glb: str,
        animations: Optional[List[str]] = None,
        blender_executable: Optional[str] = None
    ) -> str:
        """
        Auto-rigs the character mesh and generates animation clips.
        Tries Blender headless execution if available, otherwise falls back to
        the standalone glTF humanoid skinning & animation synthesizer.
        """
        anims = animations or ["walk", "idle", "run", "attack"]
        blender_bin = blender_executable or os.getenv("BLENDER_PATH", "blender")
        script_file = Path("scripts/auto_rig_and_animate.py")

        # 1. Attempt Blender Headless Pipeline if available
        if script_file.exists():
            try:
                check_cmd = [blender_bin, "--version"]
                res = subprocess.run(check_cmd, capture_output=True, timeout=5)
                if res.returncode == 0:
                    run_cmd = [
                        blender_bin,
                        "--background",
                        "--python", str(script_file),
                        "--",
                        str(input_glb),
                        str(output_glb),
                        ",".join(anims)
                    ]
                    p_res = subprocess.run(run_cmd, capture_output=True, timeout=120)
                    if p_res.returncode == 0 and Path(output_glb).exists():
                        return str(output_glb)
            except Exception as e:
                logger.info(f"Blender headless not used ({e}), falling back to standalone glTF rigger.")

        # 2. Standalone glTF Skeletal Rigging & Animation Engine Fallback
        return self._build_standalone_skinned_glb(input_glb, output_glb, anims)

    def _build_standalone_skinned_glb(
        self,
        input_glb: str,
        output_glb: str,
        animations: List[str]
    ) -> str:
        out_path = Path(output_glb)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        scene = trimesh.load(input_glb)
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if not geoms:
                raise ValueError("GLB has no geometry.")
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0].copy()
        else:
            mesh = scene.copy()

        # Center X & Z; ground to Y=0
        bounds = mesh.bounds
        center_x = (bounds[0][0] + bounds[1][0]) * 0.5
        center_z = (bounds[0][2] + bounds[1][2]) * 0.5
        min_y = bounds[0][1]
        mesh.apply_translation([-center_x, -min_y, -center_z])

        bounds = mesh.bounds
        height = max(bounds[1][1] - bounds[0][1], 1e-4)
        if height < 0.5 or height > 5.0:
            scale = 1.75 / height
            mesh.apply_scale(scale)
            bounds = mesh.bounds
            height = bounds[1][1] - bounds[0][1]

        # Ensure normals and UVs
        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
            mesh.compute_vertex_normals()

        width = max(bounds[1][0] - bounds[0][0], 1e-4)
        verts = mesh.vertices.astype(np.float32)
        normals = mesh.vertex_normals.astype(np.float32)

        # Standard 13-Bone Humanoid Hierarchy
        # Nodes: 0: Root, 1: Mesh, 2..14: Bones
        bone_defs = [
            # (name, world_pos, parent_idx, local_trans)
            ("Hips",         np.array([0.0, 0.95, 0.0], dtype=np.float32),  None, [0.0, 0.95, 0.0]),
            ("Spine",        np.array([0.0, 1.15, 0.0], dtype=np.float32),  0,    [0.0, 0.20, 0.0]),
            ("Chest",        np.array([0.0, 1.35, 0.0], dtype=np.float32),  1,    [0.0, 0.20, 0.0]),
            ("Neck",         np.array([0.0, 1.55, 0.0], dtype=np.float32),  2,    [0.0, 0.20, 0.0]),
            ("Head",         np.array([0.0, 1.68, 0.0], dtype=np.float32),  3,    [0.0, 0.13, 0.0]),
            ("UpperArm.L",   np.array([-0.22, 1.35, 0.0], dtype=np.float32), 2,   [-0.22, 0.0, 0.0]),
            ("Forearm.L",    np.array([-0.42, 1.12, 0.0], dtype=np.float32), 5,   [-0.20, -0.23, 0.0]),
            ("UpperArm.R",   np.array([0.22, 1.35, 0.0], dtype=np.float32),  2,    [0.22, 0.0, 0.0]),
            ("Forearm.R",    np.array([0.42, 1.12, 0.0], dtype=np.float32),  7,    [0.20, -0.23, 0.0]),
            ("UpperLeg.L",   np.array([-0.11, 0.88, 0.0], dtype=np.float32), 0,   [-0.11, -0.07, 0.0]),
            ("LowerLeg.L",   np.array([-0.11, 0.45, 0.0], dtype=np.float32), 9,   [0.0, -0.43, 0.0]),
            ("UpperLeg.R",   np.array([0.11, 0.88, 0.0], dtype=np.float32),  0,    [0.11, -0.07, 0.0]),
            ("LowerLeg.R",   np.array([0.11, 0.45, 0.0], dtype=np.float32),  11,   [0.0, -0.43, 0.0]),
        ]
        num_bones = len(bone_defs)
        bone_world_positions = np.array([b[1] for b in bone_defs], dtype=np.float32)

        # 1. Compute vertex weights & joints (4 nearest bones)
        num_verts = len(verts)
        # Distance matrix (num_verts, num_bones)
        dists = np.linalg.norm(verts[:, np.newaxis, :] - bone_world_positions[np.newaxis, :, :], axis=2)
        inv_dists = 1.0 / (dists + 0.04)**2.5

        # Get top 4 bone indices
        top4_joints = np.argsort(-inv_dists, axis=1)[:, :4].astype(np.uint16)
        top4_weights_raw = np.take_along_axis(inv_dists, top4_joints, axis=1).astype(np.float32)
        weight_sums = np.sum(top4_weights_raw, axis=1, keepdims=True)
        top4_weights = (top4_weights_raw / np.maximum(weight_sums, 1e-6)).astype(np.float32)

        # 2. Build Inverse Bind Matrices (4x4 column-major)
        inv_bind_matrices = []
        for _, w_pos, _, _ in bone_defs:
            # Transposed translation inverse
            m = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                -float(w_pos[0]), -float(w_pos[1]), -float(w_pos[2]), 1.0
            ]
            inv_bind_matrices.extend(m)
        inv_bind_bytes = struct.pack(f"<{num_bones * 16}f", *inv_bind_matrices)

        # 3. Vertex Attributes binary buffers
        pos_bytes = verts.tobytes()
        norm_bytes = normals.tobytes()

        # UVs
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) == num_verts:
            uvs = mesh.visual.uv.astype(np.float32)
        else:
            u = np.clip((verts[:, 0] - bounds[0][0]) / width, 0.0, 1.0)
            v = np.clip((verts[:, 1] - bounds[0][1]) / height, 0.0, 1.0)
            uvs = np.column_stack([u, v]).astype(np.float32)
        uv_bytes = uvs.tobytes()

        joints_bytes = top4_joints.tobytes()
        weights_bytes = top4_weights.tobytes()

        # Faces indices
        faces = mesh.faces.astype(np.uint32)
        indices_bytes = faces.flatten().tobytes()

        # Buffer & Accessors Assembly
        bin_chunks = []
        buffer_views = []
        accessors = []

        def add_buffer_view_and_accessor(data: bytes, component_type: int, count: int, type_str: str, target: Optional[int] = None, min_val=None, max_val=None):
            offset = sum(len(b) for b in bin_chunks)
            pad = (4 - (len(data) % 4)) % 4
            padded_data = data + b'\x00' * pad

            bv_idx = len(buffer_views)
            bv = {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(data)
            }
            if target:
                bv["target"] = target
            buffer_views.append(bv)
            bin_chunks.append(padded_data)

            acc_idx = len(accessors)
            acc = {
                "bufferView": bv_idx,
                "byteOffset": 0,
                "componentType": component_type,
                "count": count,
                "type": type_str
            }
            if min_val is not None:
                acc["min"] = min_val
            if max_val is not None:
                acc["max"] = max_val
            accessors.append(acc)
            return acc_idx

        # Accessors:
        # 0: Position
        pos_min = [float(np.min(verts[:, 0])), float(np.min(verts[:, 1])), float(np.min(verts[:, 2]))]
        pos_max = [float(np.max(verts[:, 0])), float(np.max(verts[:, 1])), float(np.max(verts[:, 2]))]
        acc_pos = add_buffer_view_and_accessor(pos_bytes, 5126, num_verts, "VEC3", target=34962, min_val=pos_min, max_val=pos_max)
        # 1: Normal
        acc_norm = add_buffer_view_and_accessor(norm_bytes, 5126, num_verts, "VEC3", target=34962)
        # 2: UV
        acc_uv = add_buffer_view_and_accessor(uv_bytes, 5126, num_verts, "VEC2", target=34962)
        # 3: Joints
        acc_joints = add_buffer_view_and_accessor(joints_bytes, 5123, num_verts, "VEC4", target=34962)
        # 4: Weights
        acc_weights = add_buffer_view_and_accessor(weights_bytes, 5126, num_verts, "VEC4", target=34962)
        # 5: Indices
        acc_indices = add_buffer_view_and_accessor(indices_bytes, 5125, len(faces) * 3, "SCALAR", target=34963)
        # 6: InverseBindMatrices
        acc_ibm = add_buffer_view_and_accessor(inv_bind_bytes, 5126, num_bones, "MAT4")

        # 4. Generate Animation Tracks (Walk, Idle, Run, Attack)
        # Nodes:
        # Node 0: Root Armature
        # Node 1: Mesh
        # Node 2..14: Bones (Bone 0 is Node 2)
        BONE_NODE_OFFSET = 2

        gltf_animations = []

        # Helper to create keyframe track sampler
        def create_sampler(timestamps: List[float], values: List[List[float]], val_type: str = "VEC4"):
            time_arr = np.array(timestamps, dtype=np.float32)
            time_bytes = time_arr.tobytes()
            acc_time = add_buffer_view_and_accessor(
                time_bytes, 5126, len(timestamps), "SCALAR",
                min_val=[float(np.min(time_arr))], max_val=[float(np.max(time_arr))]
            )

            val_arr = np.array(values, dtype=np.float32)
            val_bytes = val_arr.tobytes()
            acc_val = add_buffer_view_and_accessor(val_bytes, 5126, len(values), val_type)
            return acc_time, acc_val

        # Animation 1: Walk
        walk_channels = []
        walk_samplers = []
        walk_times = [0.0, 0.25, 0.5, 0.75, 1.0]

        # UpperLeg.L & UpperLeg.R (bones 9 & 11)
        leg_l_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, 0.38, 0.0, -0.38, 0.0]]
        leg_r_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, -0.38, 0.0, 0.38, 0.0]]
        s_in, s_out = create_sampler(walk_times, leg_l_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 9 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(walk_times, leg_r_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 11 + BONE_NODE_OFFSET, "path": "rotation"}})

        # LowerLeg.L & LowerLeg.R (bones 10 & 12)
        lower_l_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, 0.45, 0.0, 0.02, 0.0]]
        lower_r_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, 0.02, 0.0, 0.45, 0.0]]
        s_in, s_out = create_sampler(walk_times, lower_l_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 10 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(walk_times, lower_r_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 12 + BONE_NODE_OFFSET, "path": "rotation"}})

        # UpperArm.L & UpperArm.R (bones 5 & 7)
        arm_l_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, -0.30, 0.0, 0.30, 0.0]]
        arm_r_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, 0.30, 0.0, -0.30, 0.0]]
        s_in, s_out = create_sampler(walk_times, arm_l_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 5 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(walk_times, arm_r_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 7 + BONE_NODE_OFFSET, "path": "rotation"}})

        # Hips vertical bounce
        hip_trans = [[0.0, 0.95 + 0.015 * math.sin(t * math.pi * 2.0), 0.0] for t in walk_times]
        s_in, s_out = create_sampler(walk_times, hip_trans, val_type="VEC3")
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 0 + BONE_NODE_OFFSET, "path": "translation"}})

        gltf_animations.append({"name": "Walk", "samplers": walk_samplers, "channels": walk_channels})

        # Animation 2: Idle
        idle_channels = []
        idle_samplers = []
        idle_times = [0.0, 1.0, 2.0]
        chest_idle = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, 0.035, 0.0]]
        s_in, s_out = create_sampler(idle_times, chest_idle)
        idle_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        idle_channels.append({"sampler": len(idle_samplers) - 1, "target": {"node": 2 + BONE_NODE_OFFSET, "path": "rotation"}})

        hips_idle = [[0.0, 0.95 + 0.005 * math.sin(t * math.pi), 0.0] for t in idle_times]
        s_in, s_out = create_sampler(idle_times, hips_idle, val_type="VEC3")
        idle_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        idle_channels.append({"sampler": len(idle_samplers) - 1, "target": {"node": 0 + BONE_NODE_OFFSET, "path": "translation"}})

        gltf_animations.append({"name": "Idle", "samplers": idle_samplers, "channels": idle_channels})

        # Animation 3: Run
        run_channels = []
        run_samplers = []
        run_times = [0.0, 0.15, 0.30, 0.45, 0.60]

        spine_run = [_axis_angle_to_quat((1, 0, 0), 0.22) for _ in run_times]
        s_in, s_out = create_sampler(run_times, spine_run)
        run_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        run_channels.append({"sampler": len(run_samplers) - 1, "target": {"node": 1 + BONE_NODE_OFFSET, "path": "rotation"}})

        leg_l_run = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, 0.65, 0.0, -0.65, 0.0]]
        leg_r_run = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, -0.65, 0.0, 0.65, 0.0]]
        s_in, s_out = create_sampler(run_times, leg_l_run)
        run_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        run_channels.append({"sampler": len(run_samplers) - 1, "target": {"node": 9 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(run_times, leg_r_run)
        run_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        run_channels.append({"sampler": len(run_samplers) - 1, "target": {"node": 11 + BONE_NODE_OFFSET, "path": "rotation"}})

        gltf_animations.append({"name": "Run", "samplers": run_samplers, "channels": run_channels})

        # Animation 4: Attack
        attack_channels = []
        attack_samplers = []
        attack_times = [0.0, 0.4, 0.75, 1.2]

        chest_atk = [_axis_angle_to_quat((0, 1, 0), a) for a in [0.0, -0.35, 0.45, 0.0]]
        s_in, s_out = create_sampler(attack_times, chest_atk)
        attack_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        attack_channels.append({"sampler": len(attack_samplers) - 1, "target": {"node": 2 + BONE_NODE_OFFSET, "path": "rotation"}})

        arm_r_atk = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, 0.85, -0.85, 0.0]]
        s_in, s_out = create_sampler(attack_times, arm_r_atk)
        attack_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        attack_channels.append({"sampler": len(attack_samplers) - 1, "target": {"node": 7 + BONE_NODE_OFFSET, "path": "rotation"}})

        gltf_animations.append({"name": "Attack", "samplers": attack_samplers, "channels": attack_channels})

        # 5. Build glTF Nodes
        nodes = []
        # Node 0: Root Scene Object
        nodes.append({
            "name": "Armature_Root",
            "children": [1, 2]  # Mesh + Hips Bone
        })

        # Node 1: Skinned Mesh
        nodes.append({
            "name": "Character_Mesh",
            "mesh": 0,
            "skin": 0
        })

        # Bone Nodes 2 to 14
        # Precompute children indices for each bone
        children_map = {i: [] for i in range(num_bones)}
        for b_idx, (_, _, p_idx, _) in enumerate(bone_defs):
            if p_idx is not None:
                children_map[p_idx].append(b_idx + BONE_NODE_OFFSET)

        for b_idx, (b_name, _, _, l_trans) in enumerate(bone_defs):
            b_node = {
                "name": b_name,
                "translation": l_trans
            }
            if children_map[b_idx]:
                b_node["children"] = children_map[b_idx]
            nodes.append(b_node)

        # 6. Final glTF Structure Assembly
        bin_payload = b"".join(bin_chunks)

        gltf_dict = {
            "asset": {"version": "2.0", "generator": "Render3D Character Rigging Engine"},
            "scene": 0,
            "scenes": [{"name": "DefaultScene", "nodes": [0]}],
            "nodes": nodes,
            "meshes": [{
                "name": "Character_Mesh",
                "primitives": [{
                    "attributes": {
                        "POSITION": acc_pos,
                        "NORMAL": acc_norm,
                        "TEXCOORD_0": acc_uv,
                        "JOINTS_0": acc_joints,
                        "WEIGHTS_0": acc_weights
                    },
                    "indices": acc_indices,
                    "mode": 4
                }]
            }],
            "skins": [{
                "name": "Humanoid_Skin",
                "inverseBindMatrices": acc_ibm,
                "joints": [i + BONE_NODE_OFFSET for i in range(num_bones)]
            }],
            "animations": gltf_animations,
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(bin_payload)}]
        }

        # Pack GLB binary format
        json_bytes = json.dumps(gltf_dict, separators=(',', ':')).encode("utf-8")
        pad_json = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b' ' * pad_json

        pad_bin = (4 - (len(bin_payload) % 4)) % 4
        bin_payload += b'\x00' * pad_bin

        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_payload)
        glb_header = struct.pack("<III", 0x46546C67, 2, total_length)
        chunk0_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # 'JSON'
        chunk1_header = struct.pack("<II", len(bin_payload), 0x004E4942)  # 'BIN\0'

        final_glb_data = glb_header + chunk0_header + json_bytes + chunk1_header + bin_payload
        out_path.write_bytes(final_glb_data)
        logger.info(f"Built animated GLB with {len(gltf_animations)} clips: {out_path}")
        return str(out_path)
