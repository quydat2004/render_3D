import os
import io
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

def _euler_xyz_to_quat(rx: float, ry: float, rz: float) -> List[float]:
    """Converts Euler angles (rx, ry, rz in radians) into a normalized glTF quaternion [x, y, z, w]."""
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)

    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    qw = cx * cy * cz - sx * sy * sz
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > 1e-7:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return [float(qx), float(qy), float(qz), float(qw)]

class CharacterRiggingEngine:
    def __init__(self):
        pass

    def analyze_character_proportions(self, verts: np.ndarray) -> Dict[str, Any]:
        """
        Detects morphological proportions (Chibi / SD vs standard adult humanoid),
        close-to-body arm posture, and pre-bent knees so bone placement, skinning envelopes,
        and animation swing amplitudes adapt without joint distortion or self-intersection.
        """
        y_min = float(np.min(verts[:, 1]))
        y_max = float(np.max(verts[:, 1]))
        total_h = max(y_max - y_min, 1e-4)

        # Scan horizontal widths from 0.28 H to 0.86 H to find neck constriction
        slices = 44
        y_steps = np.linspace(y_min + total_h * 0.30, y_min + total_h * 0.86, slices)
        slice_h = total_h / slices
        widths = []
        for y_val in y_steps:
            m = np.abs(verts[:, 1] - y_val) < slice_h
            widths.append(float(np.max(verts[m, 0]) - np.min(verts[m, 0])) if np.any(m) else 1.0)
        widths_arr = np.array(widths)

        # Search upper 65% of scanned range for neck constriction
        upper_start = int(slices * 0.25)
        neck_idx = upper_start + int(np.argmin(widths_arr[upper_start:]))
        y_neck = float(np.clip(y_steps[neck_idx], y_min + total_h * 0.42, y_min + total_h * 0.86))

        head_m = verts[:, 1] > y_neck
        head_w = float(np.max(verts[head_m, 0]) - np.min(verts[head_m, 0])) if np.any(head_m) else 0.3
        mid_m = (verts[:, 1] > y_min + total_h * 0.35) & (verts[:, 1] <= y_neck)
        mid_w = float(np.max(verts[mid_m, 0]) - np.min(verts[mid_m, 0])) if np.any(mid_m) else 0.4

        head_height_ratio = (y_max - y_neck) / total_h
        is_chibi = bool(head_height_ratio > 0.33 or (head_w / max(mid_w, 1e-4)) > 0.82)

        # Detect close-to-body arms: ratio of total width at waist/elbow vs shoulder/head width
        waist_m = np.abs(verts[:, 1] - (y_min + (y_neck - y_min) * 0.55)) < slice_h * 2.0
        waist_span = float(np.max(verts[waist_m, 0]) - np.min(verts[waist_m, 0])) if np.any(waist_m) else mid_w
        arm_close_to_body = bool(waist_span < total_h * 0.42 or is_chibi)

        # Detect pre-bent knees: compare anterior Z centroid at knee height vs hip-to-ankle line
        y_hip_est = y_min + (y_neck - y_min) * (0.42 if is_chibi else 0.46)
        knee_y_est = y_min + (y_hip_est - y_min) * 0.50
        knee_m = np.abs(verts[:, 1] - knee_y_est) < slice_h * 1.8
        ankle_m = verts[:, 1] < (y_min + total_h * 0.14)
        hip_m = np.abs(verts[:, 1] - y_hip_est) < slice_h * 1.8

        hip_z = float(np.median(verts[hip_m, 2])) if np.any(hip_m) else 0.0
        ankle_z = float(np.median(verts[ankle_m, 2])) if np.any(ankle_m) else 0.0
        knee_z = float(np.median(verts[knee_m, 2])) if np.any(knee_m) else 0.0
        knee_forward_offset = knee_z - 0.5 * (hip_z + ankle_z)
        knees_pre_bent = bool(knee_forward_offset > total_h * 0.012)

        return {
            "is_chibi": is_chibi,
            "head_height_ratio": float(head_height_ratio),
            "arm_close_to_body": arm_close_to_body,
            "knees_pre_bent": knees_pre_bent,
            "knee_forward_offset": float(knee_forward_offset),
            "y_neck": float(y_neck),
            "y_hip": float(y_hip_est),
            "knee_z": float(knee_z)
        }

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
            named_geoms = [
                (name, g.copy())
                for name, g in scene.geometry.items()
                if hasattr(g, "vertices") and len(g.vertices) > 0 and len(g.faces) > 0
            ]
            if not named_geoms:
                raise ValueError("GLB has no geometry.")
        else:
            if len(scene.vertices) == 0 or len(scene.faces) == 0:
                raise ValueError("GLB has no geometry.")
            named_geoms = [("Character_Mesh", scene.copy())]

        # Compute unified bounds across all layers; center X & Z; ground to Y=0
        all_v_init = np.vstack([g.vertices for _, g in named_geoms])
        b_min = np.min(all_v_init, axis=0)
        b_max = np.max(all_v_init, axis=0)
        center_x = float((b_min[0] + b_max[0]) * 0.5)
        center_z = float((b_min[2] + b_max[2]) * 0.5)
        min_y = float(b_min[1])
        for _, g in named_geoms:
            g.apply_translation([-center_x, -min_y, -center_z])

        height = max(float(b_max[1] - b_min[1]), 1e-4)
        if height < 0.5 or height > 5.0:
            scale = 1.75 / height
            for _, g in named_geoms:
                g.apply_scale(scale)

        for _, g in named_geoms:
            if g.vertex_normals is None or len(g.vertex_normals) != len(g.vertices):
                g.compute_vertex_normals()

        # Use body layer for anatomical silhouette scanning when available, otherwise combined mesh
        body_candidates = [g for name, g in named_geoms if "body" in str(name).lower()]
        ref_verts = (
            body_candidates[0].vertices.astype(np.float32)
            if body_candidates
            else np.vstack([g.vertices for _, g in named_geoms]).astype(np.float32)
        )
        bounds = np.array([np.min(ref_verts, axis=0), np.max(ref_verts, axis=0)], dtype=np.float32)
        verts = ref_verts

        # Anatomical Silhouette & Cross-Section Scanning for Chibi & Adult Proportions
        props = self.analyze_character_proportions(ref_verts)
        is_chibi = props["is_chibi"]
        arm_close_to_body = props["arm_close_to_body"]
        knees_pre_bent = props["knees_pre_bent"]
        knee_forward_offset = props["knee_forward_offset"]
        knee_z_centroid = props["knee_z"]

        y_min = float(bounds[0][1])
        y_max = float(bounds[1][1])
        total_h = max(y_max - y_min, 1e-4)

        slices = 40
        y_steps = np.linspace(y_min + total_h * 0.35, y_min + total_h * 0.88, slices)
        slice_h = total_h / slices
        widths = []
        for y_val in y_steps:
            m = np.abs(verts[:, 1] - y_val) < slice_h
            if np.any(m):
                w = float(np.max(verts[m, 0]) - np.min(verts[m, 0]))
            else:
                w = 1.0
            widths.append(w)
        widths = np.array(widths)

        # Neck is local minimum width in upper body
        neck_idx = int(np.argmin(widths))
        y_neck = float(y_steps[neck_idx])
        y_neck = float(np.clip(y_neck, y_min + total_h * 0.42, y_min + total_h * 0.86))

        # Head center
        y_head = float(y_max - (y_max - y_neck) * 0.42)

        # Hips / Pelvis detection: lowest level where leg gap closes
        y_hip_search = np.linspace(y_min + total_h * 0.12, y_neck * 0.80, 20)
        y_hip = y_min + (y_neck - y_min) * (0.42 if is_chibi else 0.45)
        for y_h in y_hip_search:
            m = np.abs(verts[:, 1] - y_h) < slice_h
            if np.any(m):
                if np.any(np.abs(verts[m, 0]) < 0.03):
                    y_hip = float(y_h)
                    break

        y_spine = y_hip + (y_neck - y_hip) * 0.32
        y_chest = y_hip + (y_neck - y_hip) * 0.70

        chest_m = np.abs(verts[:, 1] - y_chest) < slice_h * 1.5
        torso_w = float(np.max(verts[chest_m, 0]) - np.min(verts[chest_m, 0])) if np.any(chest_m) else 0.4
        sh_x = max(torso_w * 0.45, 0.10)

        max_x = float(np.max(verts[:, 0]))
        elbow_x = max(max_x * (0.68 if arm_close_to_body else 0.75), sh_x * 1.25)
        elbow_y = y_chest - (y_chest - y_hip) * (0.46 if is_chibi else 0.42)

        leg_x = max(sh_x * 0.52, 0.08)
        knee_y = y_min + (y_hip - y_min) * (0.45 if is_chibi else 0.50)

        # Scan actual foot centroids to avoid inward twisting of ankles & feet
        feet_m = verts[:, 1] < (y_min + total_h * 0.16)
        left_feet = verts[feet_m & (verts[:, 0] < -0.01)]
        right_feet = verts[feet_m & (verts[:, 0] > 0.01)]
        true_leg_l_x = float(np.median(left_feet[:, 0])) if len(left_feet) > 5 else -leg_x
        true_leg_r_x = float(np.median(right_feet[:, 0])) if len(right_feet) > 5 else leg_x
        true_feet_z = float(np.median(verts[feet_m, 2])) if np.any(feet_m) else 0.0
        # Position knee bone at actual pre-bent knee centroid Z to prevent joint collapse or mesh tearing
        knee_z = float(knee_z_centroid) if (knees_pre_bent or abs(knee_forward_offset) > 0.01) else true_feet_z

        # Dynamic Humanoid 13-Bone Hierarchy
        bone_defs = [
            ("Hips",       np.array([0.0, y_hip, 0.0], dtype=np.float32),          None, [0.0, y_hip, 0.0]),
            ("Spine",      np.array([0.0, y_spine, 0.0], dtype=np.float32),        0,    [0.0, y_spine - y_hip, 0.0]),
            ("Chest",      np.array([0.0, y_chest, 0.0], dtype=np.float32),        1,    [0.0, y_chest - y_spine, 0.0]),
            ("Neck",       np.array([0.0, y_neck, 0.0], dtype=np.float32),         2,    [0.0, y_neck - y_chest, 0.0]),
            ("Head",       np.array([0.0, y_head, 0.0], dtype=np.float32),         3,    [0.0, y_head - y_neck, 0.0]),
            ("UpperArm.L", np.array([-sh_x, y_chest, 0.0], dtype=np.float32),       2,    [-sh_x, 0.0, 0.0]),
            ("Forearm.L",  np.array([-elbow_x, elbow_y, 0.0], dtype=np.float32),    5,    [-(elbow_x - sh_x), elbow_y - y_chest, 0.0]),
            ("UpperArm.R", np.array([sh_x, y_chest, 0.0], dtype=np.float32),        2,    [sh_x, 0.0, 0.0]),
            ("Forearm.R",  np.array([elbow_x, elbow_y, 0.0], dtype=np.float32),     7,    [elbow_x - sh_x, elbow_y - y_chest, 0.0]),
            ("UpperLeg.L", np.array([true_leg_l_x, y_hip - 0.04, true_feet_z], dtype=np.float32), 0,[true_leg_l_x, -0.04, true_feet_z]),
            ("LowerLeg.L", np.array([true_leg_l_x, knee_y, knee_z], dtype=np.float32),            9,[0.0, knee_y - (y_hip - 0.04), knee_z - true_feet_z]),
            ("UpperLeg.R", np.array([true_leg_r_x, y_hip - 0.04, true_feet_z], dtype=np.float32), 0,[true_leg_r_x, -0.04, true_feet_z]),
            ("LowerLeg.R", np.array([true_leg_r_x, knee_y, knee_z], dtype=np.float32),           11,[0.0, knee_y - (y_hip - 0.04), knee_z - true_feet_z]),
        ]
        num_bones = len(bone_defs)
        bone_world_positions = np.array([b[1] for b in bone_defs], dtype=np.float32)

        # 1. Helper to compute vertex skinning weights with Anatomical Envelope Protection
        def compute_skin_weights(g_verts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            dists = np.linalg.norm(g_verts[:, np.newaxis, :] - bone_world_positions[np.newaxis, :, :], axis=2)

            # Envelope Penalties to prevent cross-limb stretching (arms pulling hips or legs pulling chest)
            below_hip = g_verts[:, 1] < (y_hip - 0.04)
            dists[below_hip, 5] += 10.0
            dists[below_hip, 6] += 10.0
            dists[below_hip, 7] += 10.0
            dists[below_hip, 8] += 10.0

            above_chest = g_verts[:, 1] > (y_chest + 0.02)
            dists[above_chest, 9] += 10.0
            dists[above_chest, 10] += 10.0
            dists[above_chest, 11] += 10.0
            dists[above_chest, 12] += 10.0

            below_neck = g_verts[:, 1] < (y_neck - 0.04)
            dists[below_neck, 4] += 10.0

            # Vertices above the neck belong to the head. Lock everything above the neck to Neck and Head bones.
            above_neck = g_verts[:, 1] > y_neck
            for b in range(num_bones):
                if b not in (3, 4):
                    dists[above_neck, b] += 10.0

            # Torso vs Arm separation: prevent arms pulling the torso/waist when arms are close to body
            torso_half_w = max(torso_w * 0.48, 0.12)
            is_torso_v = (g_verts[:, 1] >= (y_hip - 0.02)) & (g_verts[:, 1] <= (y_neck - 0.02)) & (np.abs(g_verts[:, 0]) < torso_half_w)
            dists[is_torso_v, 5] += 15.0  # UpperArm.L
            dists[is_torso_v, 6] += 15.0  # Forearm.L
            dists[is_torso_v, 7] += 15.0  # UpperArm.R
            dists[is_torso_v, 8] += 15.0  # Forearm.R

            # Outer arm vertices shouldn't be bound to spine or chest bones
            is_outer_arm = (g_verts[:, 1] >= (y_hip - 0.10)) & (g_verts[:, 1] <= y_neck) & (np.abs(g_verts[:, 0]) >= (torso_half_w + 0.04))
            dists[is_outer_arm, 0] += 15.0  # Hips
            dists[is_outer_arm, 1] += 15.0  # Spine
            dists[is_outer_arm, 2] += 12.0  # Chest

            # Lateral Left vs Right limb separation: prevent left limb bones pulling right limbs and vice versa
            is_right_v = g_verts[:, 0] > 0.015
            is_left_v = g_verts[:, 0] < -0.015
            dists[is_right_v, 5] += 20.0  # UpperArm.L
            dists[is_right_v, 6] += 20.0  # Forearm.L
            dists[is_right_v, 9] += 20.0  # UpperLeg.L
            dists[is_right_v, 10] += 20.0 # LowerLeg.L

            dists[is_left_v, 7] += 20.0   # UpperArm.R
            dists[is_left_v, 8] += 20.0   # Forearm.R
            dists[is_left_v, 11] += 20.0  # UpperLeg.R
            dists[is_left_v, 12] += 20.0  # LowerLeg.R

            inv_dists = 1.0 / (dists + 0.035)**2.5
            top4_j = np.argsort(-inv_dists, axis=1)[:, :4].astype(np.uint16)
            top4_w_raw = np.take_along_axis(inv_dists, top4_j, axis=1).astype(np.float32)
            top4_w_raw[top4_w_raw < 0.04] = 0.0
            w_sums = np.sum(top4_w_raw, axis=1, keepdims=True)
            top4_w = (top4_w_raw / np.maximum(w_sums, 1e-6)).astype(np.float32)
            return top4_j, top4_w

            inv_dists = 1.0 / (dists + 0.035)**2.5
            top4_j = np.argsort(-inv_dists, axis=1)[:, :4].astype(np.uint16)
            top4_w_raw = np.take_along_axis(inv_dists, top4_j, axis=1).astype(np.float32)
            top4_w_raw[top4_w_raw < 0.04] = 0.0
            w_sums = np.sum(top4_w_raw, axis=1, keepdims=True)
            top4_w = (top4_w_raw / np.maximum(w_sums, 1e-6)).astype(np.float32)
            return top4_j, top4_w

        # 2. Build Inverse Bind Matrices (4x4 column-major)
        inv_bind_matrices = []
        for _, w_pos, _, _ in bone_defs:
            m = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                -float(w_pos[0]), -float(w_pos[1]), -float(w_pos[2]), 1.0
            ]
            inv_bind_matrices.extend(m)
        inv_bind_bytes = struct.pack(f"<{num_bones * 16}f", *inv_bind_matrices)

        # Buffer & Accessors Assembly
        bin_chunks = []
        buffer_views = []
        accessors = []
        gltf_images = []
        gltf_textures = []
        gltf_materials = []
        mesh_primitives = []

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

        def embed_pil_texture(pil_img, keep_alpha: bool = False) -> Optional[int]:
            try:
                buf = io.BytesIO()
                mode = "RGBA" if (keep_alpha and pil_img.mode == "RGBA") else "RGB"
                pil_img.convert(mode).save(buf, format="PNG")
                img_bytes = buf.getvalue()

                offset = sum(len(b) for b in bin_chunks)
                pad = (4 - (len(img_bytes) % 4)) % 4
                padded_img_bytes = img_bytes + b'\x00' * pad

                bv_idx = len(buffer_views)
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": offset,
                    "byteLength": len(img_bytes)
                })
                bin_chunks.append(padded_img_bytes)

                img_idx = len(gltf_images)
                gltf_images.append({
                    "bufferView": bv_idx,
                    "mimeType": "image/png"
                })
                tex_idx = len(gltf_textures)
                gltf_textures.append({
                    "sampler": 0,
                    "source": img_idx
                })
                return tex_idx
            except Exception as e:
                logger.warning(f"Could not encode texture image ({e}).")
                return None

        # 3. Build Primitives & Materials per Geometry Layer
        for geom_name, g in named_geoms:
            g_verts = g.vertices.astype(np.float32)
            g_normals = g.vertex_normals.astype(np.float32)
            n_v = len(g_verts)
            top4_joints, top4_weights = compute_skin_weights(g_verts)

            if hasattr(g.visual, "uv") and g.visual.uv is not None and len(g.visual.uv) == n_v:
                uvs = g.visual.uv.astype(np.float32).copy()
                uvs[:, 1] = 1.0 - uvs[:, 1]
            else:
                uvs = np.zeros((n_v, 2), dtype=np.float32)

            pos_min = [float(np.min(g_verts[:, 0])), float(np.min(g_verts[:, 1])), float(np.min(g_verts[:, 2]))]
            pos_max = [float(np.max(g_verts[:, 0])), float(np.max(g_verts[:, 1])), float(np.max(g_verts[:, 2]))]
            acc_pos = add_buffer_view_and_accessor(g_verts.tobytes(), 5126, n_v, "VEC3", target=34962, min_val=pos_min, max_val=pos_max)
            acc_norm = add_buffer_view_and_accessor(g_normals.tobytes(), 5126, n_v, "VEC3", target=34962)
            acc_uv = add_buffer_view_and_accessor(uvs.tobytes(), 5126, n_v, "VEC2", target=34962)
            acc_joints = add_buffer_view_and_accessor(top4_joints.tobytes(), 5123, n_v, "VEC4", target=34962)
            acc_weights = add_buffer_view_and_accessor(top4_weights.tobytes(), 5126, n_v, "VEC4", target=34962)

            g_faces = g.faces.astype(np.uint32)
            acc_indices = add_buffer_view_and_accessor(g_faces.flatten().tobytes(), 5125, len(g_faces) * 3, "SCALAR", target=34963)

            prim = {
                "attributes": {
                    "POSITION": acc_pos,
                    "NORMAL": acc_norm,
                    "TEXCOORD_0": acc_uv,
                    "JOINTS_0": acc_joints,
                    "WEIGHTS_0": acc_weights
                },
                "indices": acc_indices,
                "mode": 4
            }

            if hasattr(g.visual, "kind") and g.visual.kind == "vertex" and hasattr(g.visual, "vertex_colors") and g.visual.vertex_colors is not None and len(g.visual.vertex_colors) == n_v:
                vc_float = (g.visual.vertex_colors[:, :4].astype(np.float32) / 255.0)
                acc_col = add_buffer_view_and_accessor(vc_float.tobytes(), 5126, n_v, "VEC4", target=34962)
                prim["attributes"]["COLOR_0"] = acc_col

            # Extract texture and PBR material properties for this primitive
            texture_img = None
            normal_img = None
            mat_obj = getattr(g.visual, "material", None)
            if mat_obj is not None:
                if hasattr(mat_obj, "baseColorTexture") and mat_obj.baseColorTexture is not None:
                    texture_img = mat_obj.baseColorTexture
                elif hasattr(mat_obj, "image") and mat_obj.image is not None:
                    texture_img = mat_obj.image
                if hasattr(mat_obj, "normalTexture") and mat_obj.normalTexture is not None:
                    normal_img = mat_obj.normalTexture
            if texture_img is None and hasattr(g.visual, "image") and g.visual.image is not None:
                texture_img = g.visual.image

            alpha_mode = getattr(mat_obj, "alphaMode", None) if mat_obj is not None else None
            keep_alpha = (alpha_mode in ("MASK", "BLEND")) or (texture_img is not None and getattr(texture_img, "mode", "") == "RGBA")
            tex_idx = embed_pil_texture(texture_img, keep_alpha=keep_alpha) if texture_img is not None else None
            norm_idx = embed_pil_texture(normal_img, keep_alpha=False) if normal_img is not None else None

            base_factor = getattr(mat_obj, "baseColorFactor", None) if mat_obj is not None else None
            if tex_idx is not None or base_factor is not None or mat_obj is not None:
                is_armor_geom = "armor" in str(geom_name).lower()
                raw_metal = getattr(mat_obj, "metallicFactor", None) if mat_obj is not None else None
                raw_rough = getattr(mat_obj, "roughnessFactor", None) if mat_obj is not None else None

                # Enforce matte organic surface (metalness=0.0, roughness=0.65) unless explicitly armor
                if is_armor_geom:
                    metal_val = float(raw_metal if raw_metal is not None else 0.95)
                    rough_val = float(raw_rough if raw_rough is not None else 0.18)
                else:
                    metal_val = 0.0
                    rough_val = float(raw_rough if (raw_rough is not None and 0.6 <= raw_rough <= 0.7) else 0.65)

                # Synthesize or extract zone-separated PBR MetallicRoughness map (skin, fabric, hair, shoes, teeth/eyes)
                mr_img = None
                if mat_obj is not None and hasattr(mat_obj, "metallicRoughnessTexture") and mat_obj.metallicRoughnessTexture is not None:
                    mr_img = mat_obj.metallicRoughnessTexture
                elif texture_img is not None and not is_armor_geom:
                    try:
                        from services.inference.pbr_texture_engine import PBRTextureEngine
                        _, _, _, mr_img = PBRTextureEngine().synthesize_material_zone_maps(
                            texture_img,
                            base_metalness=metal_val,
                            base_roughness=rough_val
                        )
                    except Exception as e:
                        logger.debug(f"PBR zone map synthesis skipped ({e}).")

                mr_idx = embed_pil_texture(mr_img, keep_alpha=False) if mr_img is not None else None

                pbr_info: Dict[str, Any] = {
                    "metallicFactor": metal_val,
                    "roughnessFactor": rough_val
                }
                if tex_idx is not None:
                    pbr_info["baseColorTexture"] = {"index": tex_idx}
                if mr_idx is not None:
                    pbr_info["metallicRoughnessTexture"] = {"index": mr_idx}
                if base_factor is not None:
                    bf = np.asarray(base_factor, dtype=np.float32).flatten()
                    if len(bf) >= 3:
                        if np.max(bf) > 1.0:
                            bf = bf / 255.0
                        rgba = [float(bf[0]), float(bf[1]), float(bf[2]), float(bf[3]) if len(bf) > 3 else 1.0]
                        pbr_info["baseColorFactor"] = rgba

                mat_dict: Dict[str, Any] = {
                    "name": f"{geom_name}_Material",
                    "pbrMetallicRoughness": pbr_info,
                    "doubleSided": True
                }
                if norm_idx is not None:
                    mat_dict["normalTexture"] = {"index": norm_idx}
                if mr_idx is not None:
                    mat_dict["occlusionTexture"] = {"index": mr_idx}
                if alpha_mode in ("MASK", "BLEND"):
                    mat_dict["alphaMode"] = alpha_mode
                    if alpha_mode == "MASK":
                        mat_dict["alphaCutoff"] = float(getattr(mat_obj, "alphaCutoff", None) or 0.5)

                mat_idx = len(gltf_materials)
                gltf_materials.append(mat_dict)
                prim["material"] = mat_idx

            mesh_primitives.append(prim)

        # InverseBindMatrices accessor
        acc_ibm = add_buffer_view_and_accessor(inv_bind_bytes, 5126, num_bones, "MAT4")

        # 4. Generate Animation Tracks (Walk, Idle, Run, Attack) with Chibi & Arm-Clearance Adaptation
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

        # Morphological scaling factors for Chibi & arms close to body
        leg_stride_scale = 0.68 if is_chibi else 1.0
        arm_clearance_rad = 0.12 if (arm_close_to_body or is_chibi) else 0.0

        # Animation 1: Walk
        walk_channels = []
        walk_samplers = []
        walk_times = [0.0, 0.25, 0.5, 0.75, 1.0]

        # UpperLeg.L & UpperLeg.R (bones 9 & 11)
        leg_stride_w = 0.38 * leg_stride_scale
        leg_l_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, leg_stride_w, 0.0, -leg_stride_w, 0.0]]
        leg_r_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, -leg_stride_w, 0.0, leg_stride_w, 0.0]]
        s_in, s_out = create_sampler(walk_times, leg_l_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 9 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(walk_times, leg_r_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 11 + BONE_NODE_OFFSET, "path": "rotation"}})

        # LowerLeg.L & LowerLeg.R (bones 10 & 12) - Gentle knee flex
        knee_flex = 0.18 * leg_stride_scale
        lower_l_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, knee_flex, 0.0, 0.02, 0.0]]
        lower_r_quats = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, 0.02, 0.0, knee_flex, 0.0]]
        s_in, s_out = create_sampler(walk_times, lower_l_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 10 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(walk_times, lower_r_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 12 + BONE_NODE_OFFSET, "path": "rotation"}})

        # UpperArm.L & UpperArm.R (bones 5 & 7): swing with lateral outward clearance so hands never slice through hips
        arm_swing_w = 0.28 * leg_stride_scale
        arm_l_quats = [_euler_xyz_to_quat(rx=a, ry=0.0, rz=-arm_clearance_rad) for a in [0.0, -arm_swing_w, 0.0, arm_swing_w, 0.0]]
        arm_r_quats = [_euler_xyz_to_quat(rx=a, ry=0.0, rz=arm_clearance_rad) for a in [0.0, arm_swing_w, 0.0, -arm_swing_w, 0.0]]
        s_in, s_out = create_sampler(walk_times, arm_l_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 5 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(walk_times, arm_r_quats)
        walk_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        walk_channels.append({"sampler": len(walk_samplers) - 1, "target": {"node": 7 + BONE_NODE_OFFSET, "path": "rotation"}})

        # No translation on the Hips node. Walking is limbs only: legs and arms rotate, the torso and head stay still.
        gltf_animations.append({"name": "Walk", "samplers": walk_samplers, "channels": walk_channels})

        # Animation 2: Idle
        idle_channels = []
        idle_samplers = []
        gltf_animations.append({"name": "Idle", "samplers": idle_samplers, "channels": idle_channels})

        # Animation 3: Run
        run_channels = []
        run_samplers = []
        run_times = [0.0, 0.15, 0.30, 0.45, 0.60]

        spine_run = [_axis_angle_to_quat((1, 0, 0), 0.18 * leg_stride_scale) for _ in run_times]
        s_in, s_out = create_sampler(run_times, spine_run)
        run_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        run_channels.append({"sampler": len(run_samplers) - 1, "target": {"node": 1 + BONE_NODE_OFFSET, "path": "rotation"}})

        leg_stride_run = 0.65 * leg_stride_scale
        leg_l_run = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, leg_stride_run, 0.0, -leg_stride_run, 0.0]]
        leg_r_run = [_axis_angle_to_quat((1, 0, 0), a) for a in [0.0, -leg_stride_run, 0.0, leg_stride_run, 0.0]]
        s_in, s_out = create_sampler(run_times, leg_l_run)
        run_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        run_channels.append({"sampler": len(run_samplers) - 1, "target": {"node": 9 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(run_times, leg_r_run)
        run_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        run_channels.append({"sampler": len(run_samplers) - 1, "target": {"node": 11 + BONE_NODE_OFFSET, "path": "rotation"}})

        # Run arm swing with lateral clearance
        arm_swing_run = 0.45 * leg_stride_scale
        arm_l_run = [_euler_xyz_to_quat(rx=a, ry=0.0, rz=-arm_clearance_rad * 1.3) for a in [0.0, -arm_swing_run, 0.0, arm_swing_run, 0.0]]
        arm_r_run = [_euler_xyz_to_quat(rx=a, ry=0.0, rz=arm_clearance_rad * 1.3) for a in [0.0, arm_swing_run, 0.0, -arm_swing_run, 0.0]]
        s_in, s_out = create_sampler(run_times, arm_l_run)
        run_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        run_channels.append({"sampler": len(run_samplers) - 1, "target": {"node": 5 + BONE_NODE_OFFSET, "path": "rotation"}})

        s_in, s_out = create_sampler(run_times, arm_r_run)
        run_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        run_channels.append({"sampler": len(run_samplers) - 1, "target": {"node": 7 + BONE_NODE_OFFSET, "path": "rotation"}})

        gltf_animations.append({"name": "Run", "samplers": run_samplers, "channels": run_channels})

        # Animation 4: Attack
        attack_channels = []
        attack_samplers = []
        attack_times = [0.0, 0.4, 0.75, 1.2]

        chest_atk = [_axis_angle_to_quat((0, 1, 0), a) for a in [0.0, -0.30, 0.38, 0.0]]
        s_in, s_out = create_sampler(attack_times, chest_atk)
        attack_samplers.append({"input": s_in, "output": s_out, "interpolation": "LINEAR"})
        attack_channels.append({"sampler": len(attack_samplers) - 1, "target": {"node": 2 + BONE_NODE_OFFSET, "path": "rotation"}})

        atk_pitch = 0.52 if (arm_close_to_body or is_chibi) else 0.85
        arm_r_atk = [_euler_xyz_to_quat(rx=a, ry=0.18 if abs(a) > 0.05 else 0.0, rz=arm_clearance_rad) for a in [0.0, atk_pitch, -atk_pitch * 0.7, 0.0]]
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
                "primitives": mesh_primitives
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

        if gltf_materials:
            gltf_dict["materials"] = gltf_materials
        if gltf_textures:
            gltf_dict["images"] = gltf_images
            gltf_dict["samplers"] = [{
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497
            }]
            gltf_dict["textures"] = gltf_textures

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
