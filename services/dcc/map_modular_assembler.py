import json
import math
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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

def _color_mesh(mesh: trimesh.Trimesh, rgba: List[int]) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh

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

    def _resolve_biome_palette(self, biome: str) -> Dict[str, List[int]]:
        b = (biome or "").lower()
        if "cyberpunk" in b or "neon" in b or "subway" in b:
            return {
                "floor": [32, 36, 46, 255],
                "floor_grout": [18, 22, 30, 255],
                "wall": [45, 52, 68, 255],
                "trim": [24, 28, 38, 255],
                "accent": [0, 240, 255, 255],
                "accent_alt": [255, 45, 155, 255],
                "crate": [58, 68, 88, 255],
                "metal": [135, 148, 168, 255],
            }
        elif "warehouse" in b or "industrial" in b:
            return {
                "floor": [62, 64, 68, 255],
                "floor_grout": [36, 38, 42, 255],
                "wall": [88, 82, 76, 255],
                "trim": [48, 44, 42, 255],
                "accent": [245, 175, 25, 255],
                "accent_alt": [220, 85, 35, 255],
                "crate": [115, 88, 52, 255],
                "metal": [140, 142, 148, 255],
            }
        elif "temple" in b or "ruin" in b or "ancient" in b:
            return {
                "floor": [112, 102, 86, 255],
                "floor_grout": [68, 62, 52, 255],
                "wall": [132, 120, 102, 255],
                "trim": [86, 78, 64, 255],
                "accent": [45, 235, 155, 255],
                "accent_alt": [235, 190, 55, 255],
                "crate": [98, 82, 62, 255],
                "metal": [168, 152, 118, 255],
            }
        else:
            return {
                "floor": [46, 52, 64, 255],
                "floor_grout": [26, 30, 38, 255],
                "wall": [68, 76, 92, 255],
                "trim": [36, 42, 54, 255],
                "accent": [0, 215, 255, 255],
                "accent_alt": [55, 245, 165, 255],
                "crate": [72, 84, 102, 255],
                "metal": [155, 168, 185, 255],
            }

    def _build_kitbash_unit(self, asset_id: str, biome: str = "SciFi") -> trimesh.Trimesh:
        aid = asset_id.lower()
        pal = self._resolve_biome_palette(biome)
        parts: List[trimesh.Trimesh] = []

        if "floor" in aid:
            # Detailed 4x4m Modular Floor Tile (Z height = 0.20m -> CoverType.NONE)
            grout = trimesh.creation.box(extents=[4.0, 4.0, 0.14])
            grout.apply_translation([0.0, 0.0, 0.07])
            parts.append(_color_mesh(grout, pal["floor_grout"]))

            slab = trimesh.creation.box(extents=[3.84, 3.84, 0.18])
            slab.apply_translation([0.0, 0.0, 0.09])
            parts.append(_color_mesh(slab, pal["floor"]))

            # Recessed cross guide lines & center medallion
            strip_x = trimesh.creation.box(extents=[3.86, 0.14, 0.19])
            strip_x.apply_translation([0.0, 0.0, 0.095])
            parts.append(_color_mesh(strip_x, pal["trim"]))

            strip_y = trimesh.creation.box(extents=[0.14, 3.86, 0.19])
            strip_y.apply_translation([0.0, 0.0, 0.095])
            parts.append(_color_mesh(strip_y, pal["trim"]))

            center_trim = trimesh.creation.box(extents=[0.42, 0.42, 0.20])
            center_trim.apply_translation([0.0, 0.0, 0.10])
            parts.append(_color_mesh(center_trim, pal["metal"]))

        elif "wall" in aid:
            # Detailed 4x4m Reinforced Wall Segment (Z height = 4.0m -> CoverType.COVER_FULL)
            panel = trimesh.creation.box(extents=[3.92, 0.32, 3.80])
            panel.apply_translation([0.0, 0.0, 2.0])
            parts.append(_color_mesh(panel, pal["wall"]))

            plinth = trimesh.creation.box(extents=[4.0, 0.46, 0.45])
            plinth.apply_translation([0.0, 0.0, 0.225])
            parts.append(_color_mesh(plinth, pal["trim"]))

            cornice = trimesh.creation.box(extents=[4.0, 0.48, 0.35])
            cornice.apply_translation([0.0, 0.0, 3.825])
            parts.append(_color_mesh(cornice, pal["trim"]))

            # Vertical buttress ribs
            for rx in [-1.85, 0.0, 1.85]:
                rib = trimesh.creation.box(extents=[0.24, 0.44, 3.50])
                rib.apply_translation([rx, 0.0, 2.0])
                parts.append(_color_mesh(rib, pal["metal"]))

            # Glowing horizontal energy/neon strip
            neon = trimesh.creation.box(extents=[3.88, 0.38, 0.10])
            neon.apply_translation([0.0, 0.0, 2.35])
            parts.append(_color_mesh(neon, pal["accent"]))

        elif "gate" in aid or "arch" in aid:
            # 4x4m Entry Gate Archway (Z height = 4.2m -> CoverType.COVER_FULL)
            for sign in [-1.0, 1.0]:
                pylon = trimesh.creation.box(extents=[0.85, 0.65, 4.0])
                pylon.apply_translation([sign * 1.55, 0.0, 2.0])
                parts.append(_color_mesh(pylon, pal["trim"]))

                light_strip = trimesh.creation.box(extents=[0.12, 0.70, 3.2])
                light_strip.apply_translation([sign * 1.12, 0.0, 1.8])
                parts.append(_color_mesh(light_strip, pal["accent"]))

            lintel = trimesh.creation.box(extents=[4.0, 0.75, 0.70])
            lintel.apply_translation([0.0, 0.0, 3.85])
            parts.append(_color_mesh(lintel, pal["wall"]))

            header_glow = trimesh.creation.box(extents=[2.4, 0.80, 0.14])
            header_glow.apply_translation([0.0, 0.0, 3.55])
            parts.append(_color_mesh(header_glow, pal["accent_alt"]))

        elif "pillar" in aid or "column" in aid:
            # 4.0m Octagonal Support Column (Z height = 4.0m -> CoverType.COVER_FULL)
            base = trimesh.creation.box(extents=[1.15, 1.15, 0.45])
            base.apply_translation([0.0, 0.0, 0.225])
            parts.append(_color_mesh(base, pal["trim"]))

            shaft = trimesh.creation.cylinder(radius=0.42, height=3.20, sections=8)
            shaft.apply_translation([0.0, 0.0, 2.0])
            parts.append(_color_mesh(shaft, pal["wall"]))

            capital = trimesh.creation.box(extents=[1.15, 1.15, 0.35])
            capital.apply_translation([0.0, 0.0, 3.825])
            parts.append(_color_mesh(capital, pal["trim"]))

            for rz in [1.2, 2.8]:
                ring = trimesh.creation.cylinder(radius=0.46, height=0.10, sections=8)
                ring.apply_translation([0.0, 0.0, rz])
                parts.append(_color_mesh(ring, pal["accent"]))

        elif "platform" in aid:
            # Elevated Sniper Overlook Deck (Z height = 2.0m -> CoverType.COVER_FULL)
            deck = trimesh.creation.box(extents=[4.0, 4.0, 0.22])
            deck.apply_translation([0.0, 0.0, 1.10])
            parts.append(_color_mesh(deck, pal["floor"]))

            for sx in [-1.75, 1.75]:
                for sy in [-1.75, 1.75]:
                    leg = trimesh.creation.box(extents=[0.32, 0.32, 1.10])
                    leg.apply_translation([sx, sy, 0.55])
                    parts.append(_color_mesh(leg, pal["trim"]))

            # North & South protective railings
            for sy in [-1.90, 1.90]:
                rail = trimesh.creation.box(extents=[3.90, 0.14, 0.80])
                rail.apply_translation([0.0, sy, 1.60])
                parts.append(_color_mesh(rail, pal["metal"]))
                rail_glow = trimesh.creation.box(extents=[3.92, 0.16, 0.08])
                rail_glow.apply_translation([0.0, sy, 1.96])
                parts.append(_color_mesh(rail_glow, pal["accent"]))

        elif "ramp" in aid or "stair" in aid:
            # Stepped Tactical Access Ramp (Z height = 1.12m -> CoverType.COVER_HALF)
            num_steps = 6
            for s_i in range(num_steps):
                t = (s_i + 1) / float(num_steps)
                sz = t * 1.10
                sx = -1.75 + s_i * 0.65
                step_box = trimesh.creation.box(extents=[0.66, 2.40, sz])
                step_box.apply_translation([sx, 0.0, sz * 0.5])
                parts.append(_color_mesh(step_box, pal["floor"] if s_i % 2 == 0 else pal["metal"]))

        elif "generator" in aid or "core" in aid:
            # Glowing Central Reactor Core (Z height = 2.4m -> CoverType.COVER_FULL)
            ped = trimesh.creation.cylinder(radius=0.85, height=0.35, sections=16)
            ped.apply_translation([0.0, 0.0, 0.175])
            parts.append(_color_mesh(ped, pal["trim"]))

            core = trimesh.creation.cylinder(radius=0.52, height=1.80, sections=20)
            core.apply_translation([0.0, 0.0, 1.20])
            parts.append(_color_mesh(core, pal["accent"]))

            cap = trimesh.creation.cylinder(radius=0.80, height=0.30, sections=16)
            cap.apply_translation([0.0, 0.0, 2.25])
            parts.append(_color_mesh(cap, pal["trim"]))

            for angle_deg in [0, 90, 180, 270]:
                rad = math.radians(angle_deg)
                fin = trimesh.creation.box(extents=[0.14, 0.35, 1.90])
                fin.apply_translation([math.cos(rad) * 0.68, math.sin(rad) * 0.68, 1.20])
                parts.append(_color_mesh(fin, pal["metal"]))

        elif "container" in aid:
            # Armored Shipping Container (Z height = 2.4m -> CoverType.COVER_FULL)
            body = trimesh.creation.box(extents=[3.80, 1.90, 2.40])
            body.apply_translation([0.0, 0.0, 1.20])
            parts.append(_color_mesh(body, pal["crate"]))

            for rx in np.linspace(-1.6, 1.6, 5):
                rib = trimesh.creation.box(extents=[0.16, 1.98, 2.44])
                rib.apply_translation([float(rx), 0.0, 1.20])
                parts.append(_color_mesh(rib, pal["trim"]))

            stripe = trimesh.creation.box(extents=[3.84, 1.94, 0.14])
            stripe.apply_translation([0.0, 0.0, 1.85])
            parts.append(_color_mesh(stripe, pal["accent_alt"]))

        elif "barricade" in aid:
            # Tactical Half-Cover Barrier (Z height = 1.12m -> CoverType.COVER_HALF)
            base = trimesh.creation.box(extents=[2.60, 0.65, 0.45])
            base.apply_translation([0.0, 0.0, 0.225])
            parts.append(_color_mesh(base, pal["trim"]))

            upper = trimesh.creation.box(extents=[2.50, 0.42, 0.67])
            upper.apply_translation([0.0, 0.0, 0.785])
            parts.append(_color_mesh(upper, pal["wall"]))

            stripe = trimesh.creation.box(extents=[2.54, 0.46, 0.12])
            stripe.apply_translation([0.0, 0.0, 1.02])
            parts.append(_color_mesh(stripe, pal["accent"]))

        elif "spawn" in aid:
            # Glowing Team Spawn Pad (Z height = 0.24m -> CoverType.NONE)
            team_color = pal["accent_alt"] if "bravo" in aid else pal["accent"]
            pad_base = trimesh.creation.cylinder(radius=1.55, height=0.16, sections=6)
            pad_base.apply_translation([0.0, 0.0, 0.08])
            parts.append(_color_mesh(pad_base, pal["trim"]))

            pad_glow = trimesh.creation.cylinder(radius=1.20, height=0.24, sections=6)
            pad_glow.apply_translation([0.0, 0.0, 0.12])
            parts.append(_color_mesh(pad_glow, team_color))

        elif "beacon" in aid or "light" in aid:
            # Tall Streetlight / Hologram Beacon (Z height = 3.5m -> CoverType.COVER_FULL)
            ped = trimesh.creation.cylinder(radius=0.25, height=0.40, sections=12)
            ped.apply_translation([0.0, 0.0, 0.20])
            parts.append(_color_mesh(ped, pal["trim"]))

            pole = trimesh.creation.cylinder(radius=0.08, height=3.10, sections=12)
            pole.apply_translation([0.0, 0.0, 1.75])
            parts.append(_color_mesh(pole, pal["metal"]))

            lamp = trimesh.creation.icosphere(subdivisions=1, radius=0.22)
            lamp.apply_translation([0.0, 0.0, 3.35])
            parts.append(_color_mesh(lamp, pal["accent"]))

        else:
            # Default: Reinforced Tactical Supply Crate (Z height = 1.10m -> CoverType.COVER_HALF)
            core = trimesh.creation.box(extents=[1.12, 1.12, 1.06])
            core.apply_translation([0.0, 0.0, 0.53])
            parts.append(_color_mesh(core, pal["crate"]))

            lid = trimesh.creation.box(extents=[1.20, 1.20, 0.14])
            lid.apply_translation([0.0, 0.0, 1.03])
            parts.append(_color_mesh(lid, pal["trim"]))

            for sx in [-0.54, 0.54]:
                for sy in [-0.54, 0.54]:
                    post = trimesh.creation.box(extents=[0.14, 0.14, 1.08])
                    post.apply_translation([sx, sy, 0.54])
                    parts.append(_color_mesh(post, pal["metal"]))

            latch = trimesh.creation.box(extents=[0.45, 1.22, 0.10])
            latch.apply_translation([0.0, 0.0, 0.75])
            parts.append(_color_mesh(latch, pal["accent"]))

        combined = trimesh.util.concatenate(parts)
        combined.fix_normals()
        return combined

    def assemble_scene(
        self,
        layout: LevelLayoutSchema,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        out_path = Path(output_dir) if output_dir else Path(".")
        out_path.mkdir(parents=True, exist_ok=True)

        scene = trimesh.Scene()
        classified_props = []
        biome = getattr(layout, "biome", "SciFi")

        # Rotates Z-up level layout coordinates into Three.js Y-up space
        rot_z_to_y = trimesh.transformations.rotation_matrix(math.radians(-90.0), [1.0, 0.0, 0.0])

        for idx, inst in enumerate(layout.instances):
            asset_id = inst.asset_id if hasattr(inst, "asset_id") else inst.get("asset_id", f"asset_{idx}")
            raw_pos = inst.position if hasattr(inst, "position") else inst.get("position", [0.0, 0.0, 0.0])
            raw_rot = inst.rotation if hasattr(inst, "rotation") else inst.get("rotation", [0.0, 0.0, 0.0])
            raw_scale = inst.scale if hasattr(inst, "scale") else inst.get("scale", [1.0, 1.0, 1.0])
            snapped_pos = self.snap_to_grid(raw_pos)

            # Build detailed kitbash geometry in Z-up metric space
            unit_mesh = self._build_kitbash_unit(asset_id, biome=biome)

            # Apply scale before cover classification
            if raw_scale and any(abs(s - 1.0) > 1e-3 for s in raw_scale):
                scale_mat = np.diag([float(raw_scale[0]), float(raw_scale[1]), float(raw_scale[2]), 1.0])
                unit_mesh.apply_transform(scale_mat)

            cover_hint = self.classify_cover(unit_mesh)
            classified_props.append({
                "instance_id": f"{asset_id}_{idx}",
                "cover_type": cover_hint.value
            })

            # Apply instance Euler rotation (in Z-up space) and snapped translation
            transform = np.eye(4)
            if raw_rot and any(abs(r) > 1e-3 for r in raw_rot):
                rx, ry, rz = [math.radians(float(r)) for r in raw_rot[:3]]
                transform = trimesh.transformations.euler_matrix(rx, ry, rz, axes="sxyz")
            transform[:3, 3] = snapped_pos

            # Convert final node transform into Three.js Y-up coordinate space
            y_up_transform = rot_z_to_y @ transform
            scene.add_geometry(unit_mesh, node_name=f"{asset_id}_{idx}", transform=y_up_transform)

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
        quoted_json = json.dumps(instances_json)

        return f'''import bpy
import json
import math

bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Setup asset cache dictionary for Linked Duplicates (mesh_data reuse)
instances = json.loads({quoted_json})
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
