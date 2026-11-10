import json
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import trimesh

from services.orchestrator.gemini_orchestrator import LevelLayoutSchema

class CoverType(str, Enum):
    NONE = "none"
    COVER_HALF = "cover_half"
    COVER_FULL = "cover_full"

class CoverHint(BaseModel):
    prop_id: str
    cover_type: CoverType
    height_m: float

class ModularMapAssembler:
    def __init__(self, grid_size_m: float = 2.0):
        self.grid_size_m = grid_size_m

    def snap_to_grid(self, position: List[float]) -> List[float]:
        # Snap [x, y, z] coordinates to discrete metric grid intervals
        return [
            float(round(coord / self.grid_size_m) * self.grid_size_m)
            for coord in position
        ]

    def classify_cover(self, mesh: trimesh.Trimesh) -> CoverType:
        bounds = mesh.bounds
        height = float(bounds[1][2] - bounds[0][2])

        if 0.8 <= height <= 1.3:
            return CoverType.COVER_HALF
        elif height >= 1.8:
            return CoverType.COVER_FULL
        else:
            return CoverType.NONE

    def assemble_scene(
        self,
        layout: LevelLayoutSchema,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        out_path = Path(output_dir) if output_dir else Path(".")
        out_path.mkdir(parents=True, exist_ok=True)

        scene = trimesh.Scene()
        classified_props = []

        for idx, inst in enumerate(layout.instances):
            asset_id = inst.asset_id if hasattr(inst, "asset_id") else inst.get("asset_id", f"asset_{idx}")
            raw_pos = inst.position if hasattr(inst, "position") else inst.get("position", [0.0, 0.0, 0.0])
            snapped_pos = self.snap_to_grid(raw_pos)

            # Generate or fetch standard kitbash unit
            if "floor" in asset_id:
                unit_mesh = trimesh.creation.box(extents=[4.0, 4.0, 0.2])
            elif "wall" in asset_id:
                unit_mesh = trimesh.creation.box(extents=[4.0, 0.4, 4.0])
            else:
                unit_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

            cover_hint = self.classify_cover(unit_mesh)
            classified_props.append({
                "instance_id": f"{asset_id}_{idx}",
                "cover_type": cover_hint.value
            })

            # Apply translation
            transform = np.eye(4)
            transform[:3, 3] = snapped_pos
            scene.add_geometry(unit_mesh, node_name=f"{asset_id}_{idx}", transform=transform)

        # Calculate 2D walkable area
        walkable_area = float(layout.dimensions[0] * layout.dimensions[1] * 0.75)

        # Export combined scene
        glb_output = out_path / f"{layout.level_id}.glb"
        scene.export(str(glb_output), file_type="glb")

        return {
            "success": True,
            "level_id": layout.level_id,
            "instance_count": len(layout.instances),
            "output_glb": str(glb_output),
            "classified_props": classified_props,
            "navmesh": {
                "walkable_area_m2": walkable_area,
                "walkable_slope_max_deg": 45.0,
                "status": "ready"
            }
        }

    def generate_blender_assembly_script(
        self,
        layout: LevelLayoutSchema,
        output_glb: str
    ) -> str:
        safe_out = output_glb.replace("\\", "/")
        inst_list = [i.model_dump() if hasattr(i, "model_dump") else i for i in layout.instances]
        instances_json = json.dumps(inst_list)

        return f'''import bpy
import json
import math

bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Setup asset cache dictionary for Linked Duplicates (mesh_data reuse)
instances = json.loads('{instances_json}')
mesh_cache = {{}}

# 2. Spawn and link instances
for inst in instances:
    asset_id = inst.get("asset_id", "wall_01")
    pos = inst.get("position", [0, 0, 0])
    rot = inst.get("rotation", [0, 0, 0])

    if asset_id not in mesh_cache:
        # Create base mesh data once
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        base_obj = bpy.context.active_object
        base_obj.name = f"BASE_{{asset_id}}"
        mesh_cache[asset_id] = base_obj.data
        bpy.context.collection.objects.unlink(base_obj)

    # Create linked duplicate pointing to shared mesh_data
    new_obj = bpy.data.objects.new(f"INST_{{asset_id}}", mesh_cache[asset_id])
    new_obj.location = (pos[0], pos[1], pos[2])
    new_obj.rotation_euler = (math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2]))
    bpy.context.collection.objects.link(new_obj)

# 3. Export complete level to glTF 2.0
bpy.ops.export_scene.gltf(
    filepath="{safe_out}",
    export_format='GLB',
    export_apply=True
)
'''
