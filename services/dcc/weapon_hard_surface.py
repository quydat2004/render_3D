import math
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import trimesh

class WeaponSocketType(str, Enum):
    GRIP = "grip"
    MUZZLE = "muzzle"
    OPTIC = "optic"
    MAGAZINE = "magazine"
    UNDERBARREL = "underbarrel"

class WeaponSocket(BaseModel):
    name: str = Field(description="Engine socket identifier, e.g., SOCKET_Muzzle")
    socket_type: WeaponSocketType
    position: List[float] = Field(description="[X, Y, Z] metric coordinates")
    rotation: List[float] = Field(default=[0.0, 0.0, 0.0], description="[Pitch, Yaw, Roll] in degrees")

class WeaponHardSurfaceEngine:
    def __init__(self):
        pass

    def detect_sockets(self, mesh: trimesh.Trimesh) -> List[WeaponSocket]:
        bounds = mesh.bounds
        min_pt = bounds[0]
        max_pt = bounds[1]

        center_x = float((min_pt[0] + max_pt[0]) / 2.0)
        center_y = float((min_pt[1] + max_pt[1]) / 2.0)
        center_z = float((min_pt[2] + max_pt[2]) / 2.0)

        length_y = float(max_pt[1] - min_pt[1])
        height_z = float(max_pt[2] - min_pt[2])

        sockets: List[WeaponSocket] = []

        # 1. SOCKET_Muzzle: extreme forward point (+2cm along +Y vector)
        sockets.append(WeaponSocket(
            name="SOCKET_Muzzle",
            socket_type=WeaponSocketType.MUZZLE,
            position=[center_x, float(max_pt[1] + 0.02), center_z],
            rotation=[0.0, 0.0, 0.0]
        ))

        # 2. SOCKET_Grip: lower rear quartile
        sockets.append(WeaponSocket(
            name="SOCKET_Grip",
            socket_type=WeaponSocketType.GRIP,
            position=[
                center_x,
                float(min_pt[1] + length_y * 0.25),
                float(min_pt[2] + height_z * 0.20)
            ],
            rotation=[15.0, 0.0, 0.0]
        ))

        # 3. SOCKET_Optic: highest planar surface at top
        sockets.append(WeaponSocket(
            name="SOCKET_Optic",
            socket_type=WeaponSocketType.OPTIC,
            position=[center_x, center_y, float(max_pt[2])],
            rotation=[0.0, 0.0, 0.0]
        ))

        # 4. SOCKET_Magazine: lower-middle centroid
        sockets.append(WeaponSocket(
            name="SOCKET_Magazine",
            socket_type=WeaponSocketType.MAGAZINE,
            position=[
                center_x,
                float(min_pt[1] + length_y * 0.40),
                float(min_pt[2])
            ],
            rotation=[0.0, 0.0, 0.0]
        ))

        return sockets

    def generate_collision_hull(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        # Generate convex hull for real-time physics collision (UCX)
        hull = mesh.convex_hull
        hull.fix_normals()
        return hull

    def generate_weapon_blender_script(
        self,
        mesh_path: str,
        output_glb: str
    ) -> str:
        safe_mesh = mesh_path.replace("\\", "/")
        safe_out = output_glb.replace("\\", "/")

        return f'''import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Import weapon mesh
bpy.ops.wm.obj_import(filepath="{safe_mesh}")
weapon_obj = [o for o in bpy.context.selected_objects if o.type == 'MESH'][0]
weapon_obj.name = "Weapon_Main"
bpy.context.view_layer.objects.active = weapon_obj

# 2. Hard-Surface Sharp Edges (> 35 deg)
bpy.ops.object.mode_set(mode='EDIT')
mesh = weapon_obj.data
bpy.ops.mesh.select_all(action='SELECT')
for edge in mesh.edges:
    if edge.calc_face_angle() > math.radians(35.0):
        edge.use_edge_sharp = True
bpy.ops.object.mode_set(mode='OBJECT')

# 3. Apply Weighted Normal Modifier
wn_mod = weapon_obj.modifiers.new(name="WeightedNormal", type='WEIGHTED_NORMAL')
wn_mod.weight = 50
wn_mod.keep_sharp = True

# 4. Add Sockets as Empty Objects
def add_socket(name, pos, parent_obj):
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = 'ARROWS'
    empty.location = pos
    empty.parent = parent_obj
    bpy.context.collection.objects.link(empty)
    return empty

add_socket("SOCKET_Muzzle", (0, 0.75, 0), weapon_obj)
add_socket("SOCKET_Grip", (0, -0.15, -0.1), weapon_obj)
add_socket("SOCKET_Optic", (0, 0.1, 0.15), weapon_obj)

# 5. Create UCX Collision Mesh
bpy.ops.object.select_all(action='DESELECT')
weapon_obj.select_set(True)
bpy.ops.object.duplicate()
ucx_obj = bpy.context.selected_objects[0]
ucx_obj.name = "UCX_Weapon_01"
bpy.context.view_layer.objects.active = ucx_obj

# Convex hull operator
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.convex_hull()
bpy.ops.object.mode_set(mode='OBJECT')

# 6. Export glTF 2.0 with sockets and collision
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(
    filepath="{safe_out}",
    export_format='GLB',
    export_extras=True,
    export_apply=True
)
'''
