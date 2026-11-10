import os
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import trimesh

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

        # Sort weights descending
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

# 1. Import character mesh
bpy.ops.wm.obj_import(filepath="{safe_mesh}")
mesh_obj = [o for o in bpy.context.selected_objects if o.type == 'MESH'][0]

# 2. Build Armature
armature_data = bpy.data.armatures.new("Humanoid_Armature")
armature_obj = bpy.data.objects.new("Humanoid_Rig", armature_data)
bpy.context.collection.objects.link(armature_obj)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.mode_set(mode='EDIT')

# Create Pelvis and Head bones
bones = armature_data.edit_bones
pelvis_bone = bones.new("Pelvis")
pelvis_bone.head = (0, 0, 1.0)
pelvis_bone.tail = (0, 0, 1.2)

head_bone = bones.new("Head")
head_bone.head = (0, 0, 1.6)
head_bone.tail = (0, 0, 1.8)
head_bone.parent = pelvis_bone

bpy.ops.object.mode_set(mode='OBJECT')

# 3. Parent mesh with automatic weights
mesh_obj.select_set(True)
armature_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj
bpy.ops.object.parent_set(type='ARMATURE_AUTO')

# 4. Enforce max 4 bone influences (clamp_vertex_weights)
def clamp_vertex_weights(obj, max_influences=4):
    for v in obj.data.vertices:
        groups = sorted(v.groups, key=lambda g: g.weight, reverse=True)
        if len(groups) > max_influences:
            for g in groups[max_influences:]:
                obj.vertex_groups[g.group].remove([v.index])
        # Re-normalize
        total = sum(g.weight for g in obj.data.vertices[v.index].groups)
        if total > 0:
            for g in obj.data.vertices[v.index].groups:
                g.weight = g.weight / total

clamp_vertex_weights(mesh_obj)

# 5. Export glTF with skinning
bpy.ops.export_scene.gltf(
    filepath="{safe_out}",
    export_format='GLB',
    export_skins=True,
    export_all_influences=False
)
'''
