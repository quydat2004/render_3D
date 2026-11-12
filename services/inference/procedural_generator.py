import math
from pathlib import Path
from enum import Enum
from typing import Optional, List
import numpy as np
import trimesh
from skimage import measure

from services.orchestrator.gemini_orchestrator import OrthographicConceptPrompt

def _smin(d1: np.ndarray, d2: np.ndarray, k: float = 0.08) -> np.ndarray:
    h = np.maximum(k - np.abs(d1 - d2), 0.0)
    return np.minimum(d1, d2) - (h * h) / (4.0 * k)

def _smax(d1: np.ndarray, d2: np.ndarray, k: float = 0.08) -> np.ndarray:
    h = np.maximum(k - np.abs(d1 - d2), 0.0)
    return np.maximum(d1, d2) + (h * h) / (4.0 * k)

def _sdf_sphere(p: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    return np.linalg.norm(p - c, axis=-1) - radius

def _sdf_capsule(p: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    pa = p - np.asarray(a, dtype=np.float32)
    ba = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    baba = np.dot(ba, ba)
    if baba == 0.0:
        return np.linalg.norm(pa, axis=-1) - radius
    h = np.clip(np.sum(pa * ba, axis=-1) / baba, 0.0, 1.0)
    return np.linalg.norm(pa - ba * h[..., np.newaxis], axis=-1) - radius

def _sdf_ellipsoid(p: np.ndarray, center: np.ndarray, radii: np.ndarray) -> np.ndarray:
    r = np.asarray(radii, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32)
    pn = (p - c) / r
    k0 = np.linalg.norm(pn, axis=-1)
    k1 = np.linalg.norm(pn / r, axis=-1)
    return k0 * (k0 - 1.0) / (k1 + 1e-7)

def _sdf_box(p: np.ndarray, center: np.ndarray, extents: np.ndarray) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    b = np.asarray(extents, dtype=np.float32) * 0.5
    q = np.abs(p - c) - b
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0.0)
    return outside + inside

class CharacterArchetype(str, Enum):
    GHOST_ROBED = "ghost_robed"
    WARRIOR_HUMANOID = "warrior_humanoid"
    MECHA_HEAVY = "mecha_heavy"

class WeaponArchetype(str, Enum):
    SWORD_BLADE = "sword_blade"
    RIFLE_GUN = "rifle_gun"
    HAMMER_BLUNT = "hammer_blunt"

def _paint(mesh: trimesh.Trimesh, rgba: List[int]) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh

class Procedural3DGenerator:
    def __init__(self):
        pass

    def detect_character_archetype(self, prompt: str) -> CharacterArchetype:
        p = prompt.lower()
        if any(w in p for w in ["ma", "nữ", "ghost", "witch", "spirit", "robed", "áo dài", "dress", "kinh dị"]):
            return CharacterArchetype.GHOST_ROBED
        elif any(w in p for w in ["mecha", "robot", "titan", "cyborg", "golem", "tank"]):
            return CharacterArchetype.MECHA_HEAVY
        else:
            return CharacterArchetype.WARRIOR_HUMANOID

    def detect_weapon_archetype(self, prompt: str) -> WeaponArchetype:
        p = prompt.lower()
        if any(w in p for w in ["katana", "sword", "kiếm", "đao", "blade", "dagger", "dao"]):
            return WeaponArchetype.SWORD_BLADE
        elif any(w in p for w in ["hammer", "búa", "mace", "axe", "rìu", "chùy"]):
            return WeaponArchetype.HAMMER_BLUNT
        else:
            return WeaponArchetype.RIFLE_GUN

    def generate_multi_layer_character(
        self,
        prompt: str,
        output_glb_path: Optional[str] = None,
        concept: Optional[OrthographicConceptPrompt] = None
    ) -> trimesh.Scene:
        from services.inference.multi_layer_character import (
            generate_bony_base_body,
            generate_clothing_shell,
            generate_plate_armor,
            generate_hair_cards
        )
        from services.inference.pbr_texture_engine import PBRTextureEngine, _hex_to_rgb

        archetype = self.detect_character_archetype(prompt)
        tex_engine = PBRTextureEngine()

        palette = concept.palette if concept and hasattr(concept, "palette") else None
        clothing_type = getattr(palette, "clothing_type", "robe" if "ghost" in archetype.value else "suit")
        metalness = float(getattr(palette, "metalness", 0.05))
        roughness = float(getattr(palette, "roughness", 0.8))

        tex_set = tex_engine.generate_character_textures(archetype.value, width=1024, height=1024, palette=palette)

        skin_rgb = _hex_to_rgb(getattr(palette, "skin_color_hex", "#E8BEA8"))[:3]
        hair_rgb = _hex_to_rgb(getattr(palette, "hair_color_hex", "#DC2626"))[:3]
        cloth_rgb = _hex_to_rgb(getattr(palette, "primary_cloth_hex", "#D4AF37"))[:3]

        # 1. Base Body with Dedicated Skin PBR Material (Smooth natural skin)
        body_mesh = generate_bony_base_body(archetype.value)
        pbr_body = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[c / 255.0 for c in skin_rgb] + [1.0],
            baseColorTexture=tex_set.albedo,
            metallicFactor=0.0,
            roughnessFactor=0.55
        )
        body_mesh.visual = trimesh.visual.TextureVisuals(uv=body_mesh.visual.uv, material=pbr_body)

        # 2. Outfit Geometry: Solid Plate Armor OR Flowing Fabric Shell
        if clothing_type in ["armor", "knight", "mecha"] or metalness > 0.6:
            outfit_mesh = generate_plate_armor(archetype.value)
            pbr_outfit = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[c / 255.0 for c in cloth_rgb] + [1.0],
                roughnessFactor=0.18,
                metallicFactor=0.95,
                doubleSided=True
            )
            outfit_name = "Mesh_Armor"
        else:
            outfit_mesh = generate_clothing_shell(archetype.value, clothing_type=clothing_type)
            pbr_outfit = trimesh.visual.material.PBRMaterial(
                baseColorTexture=tex_set.alpha_cloth,
                normalTexture=tex_set.normal,
                roughnessFactor=roughness,
                metallicFactor=metalness,
                doubleSided=True,
                alphaMode="MASK",
                alphaCutoff=0.5
            )
            outfit_name = "Mesh_Clothing"

        outfit_mesh.visual = trimesh.visual.TextureVisuals(uv=outfit_mesh.visual.uv, material=pbr_outfit)

        # 3. Hair Cards with Tinted Alpha Strands
        hair_mesh = generate_hair_cards(head_center=np.array([0.0, 0.0, 1.62], dtype=np.float32), num_cards=44)
        pbr_hair = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[c / 255.0 for c in hair_rgb] + [1.0],
            baseColorTexture=tex_set.alpha_hair,
            roughnessFactor=0.35,
            metallicFactor=0.05,
            doubleSided=True,
            alphaMode="MASK",
            alphaCutoff=0.5
        )
        hair_mesh.visual = trimesh.visual.TextureVisuals(uv=hair_mesh.visual.uv, material=pbr_hair)

        # Coordinate transformation Z-up to Y-up
        rot_z_to_y = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        body_mesh.apply_transform(rot_z_to_y)
        outfit_mesh.apply_transform(rot_z_to_y)
        hair_mesh.apply_transform(rot_z_to_y)

        # Ground base to Y=0
        ground_y = float(body_mesh.bounds[0][1])
        body_mesh.apply_translation([0.0, -ground_y, 0.0])
        outfit_mesh.apply_translation([0.0, -ground_y, 0.0])
        hair_mesh.apply_translation([0.0, -ground_y, 0.0])

        body_mesh.fix_normals()
        outfit_mesh.fix_normals()
        hair_mesh.fix_normals()

        # Build composite Scene
        scene = trimesh.Scene()
        scene.add_geometry(body_mesh, node_name="Mesh_Body", geom_name="Mesh_Body")
        scene.add_geometry(outfit_mesh, node_name=outfit_name, geom_name=outfit_name)
        scene.add_geometry(hair_mesh, node_name="Mesh_HairCards", geom_name="Mesh_HairCards")

        if output_glb_path:
            out_file = Path(output_glb_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            scene.export(str(out_file), file_type="glb")

        return scene

    def generate_character_mesh(
        self,
        prompt: str,
        concept: Optional[OrthographicConceptPrompt] = None
    ) -> trimesh.Trimesh:
        archetype = self.detect_character_archetype(prompt)

        # Color definition
        if archetype == CharacterArchetype.GHOST_ROBED:
            c_skin = [235, 240, 245, 255]     # Pale ghostly white/cyan
            c_hair = [15, 16, 22, 255]        # Raven black hair
            c_eye_iris = [210, 20, 25, 255]   # Horror red glowing eyes
            c_lips = [140, 35, 45, 255]       # Dark blood crimson lips
            c_clothes = [245, 248, 252, 255]  # Burial white shroud/dress
            c_pants = c_clothes
            c_shoes = c_clothes
        elif archetype == CharacterArchetype.MECHA_HEAVY:
            c_skin = [110, 120, 135, 255]     # Steel chassis
            c_hair = [40, 45, 55, 255]        # Carbon plating
            c_eye_iris = [0, 240, 255, 255]   # Glowing cyan optics
            c_lips = [30, 35, 40, 255]        # Grille
            c_clothes = [50, 60, 75, 255]     # Heavy armor plating
            c_pants = [40, 48, 62, 255]
            c_shoes = [25, 30, 40, 255]
        else:
            c_skin = [230, 190, 165, 255]     # Natural human skin
            c_hair = [35, 28, 24, 255]        # Dark brown/black hair
            c_eye_iris = [30, 30, 35, 255]    # Natural dark iris
            c_lips = [185, 75, 80, 255]       # Rose lips
            c_clothes = [40, 48, 62, 255]     # Tactical navy/carbon suit
            c_pants = [30, 36, 48, 255]
            c_shoes = [20, 22, 26, 255]

        # Voxel grid parameters (High resolution for detailed geometry)
        res_x, res_y, res_z = 84, 64, 130
        min_x, max_x = -0.55, 0.55
        min_y, max_y = -0.35, 0.35
        min_z, max_z = 0.0, 1.90

        vx = np.linspace(min_x, max_x, res_x, dtype=np.float32)
        vy = np.linspace(min_y, max_y, res_y, dtype=np.float32)
        vz = np.linspace(min_z, max_z, res_z, dtype=np.float32)
        X, Y, Z = np.meshgrid(vx, vy, vz, indexing='ij')
        P = np.stack([X, Y, Z], axis=-1)

        # 1. Head Cranium, Jaw, Chin & Ears
        d_cranium = _sdf_ellipsoid(P, np.array([0.0, -0.01, 1.63], dtype=np.float32), np.array([0.115, 0.125, 0.13], dtype=np.float32))
        d_jaw = _sdf_ellipsoid(P, np.array([0.0, 0.04, 1.54], dtype=np.float32), np.array([0.075, 0.09, 0.08], dtype=np.float32))
        d_head = _smin(d_cranium, d_jaw, k=0.06)

        d_chin = _sdf_sphere(P, np.array([0.0, 0.095, 1.50], dtype=np.float32), radius=0.028)
        d_head = _smin(d_head, d_chin, k=0.03)

        d_ear_l = _sdf_ellipsoid(P, np.array([-0.118, -0.01, 1.60], dtype=np.float32), np.array([0.015, 0.03, 0.045], dtype=np.float32))
        d_ear_r = _sdf_ellipsoid(P, np.array([0.118, -0.01, 1.60], dtype=np.float32), np.array([0.015, 0.03, 0.045], dtype=np.float32))
        d_head = _smin(d_head, d_ear_l, k=0.02)
        d_head = _smin(d_head, d_ear_r, k=0.02)

        # Carved Eye Sockets & Protruding Eyeballs
        d_sock_l = _sdf_sphere(P, np.array([-0.045, 0.088, 1.625], dtype=np.float32), radius=0.028)
        d_sock_r = _sdf_sphere(P, np.array([0.045, 0.088, 1.625], dtype=np.float32), radius=0.028)
        d_head = _smax(d_head, -d_sock_l, k=0.02)
        d_head = _smax(d_head, -d_sock_r, k=0.02)

        d_eye_l = _sdf_sphere(P, np.array([-0.045, 0.084, 1.625], dtype=np.float32), radius=0.018)
        d_eye_r = _sdf_sphere(P, np.array([0.045, 0.084, 1.625], dtype=np.float32), radius=0.018)
        d_head = _smin(d_head, d_eye_l, k=0.01)
        d_head = _smin(d_head, d_eye_r, k=0.01)

        # Chiseled Nose Bridge + Tip
        d_nose_b = _sdf_capsule(P, np.array([0.0, 0.08, 1.64], dtype=np.float32), np.array([0.0, 0.125, 1.58], dtype=np.float32), radius=0.014)
        d_nose_tip = _sdf_sphere(P, np.array([0.0, 0.13, 1.575], dtype=np.float32), radius=0.018)
        d_head = _smin(d_head, _smin(d_nose_b, d_nose_tip, k=0.015), k=0.02)

        # Upper and Lower Lips
        d_lip_u = _sdf_ellipsoid(P, np.array([0.0, 0.108, 1.545], dtype=np.float32), np.array([0.028, 0.012, 0.009], dtype=np.float32))
        d_lip_l = _sdf_ellipsoid(P, np.array([0.0, 0.104, 1.530], dtype=np.float32), np.array([0.024, 0.012, 0.010], dtype=np.float32))
        d_head = _smin(d_head, d_lip_u, k=0.015)
        d_head = _smin(d_head, d_lip_l, k=0.015)

        # 2. Neck & Clavicle (Collarbone)
        d_neck = _sdf_capsule(P, np.array([0.0, 0.0, 1.40], dtype=np.float32), np.array([0.0, 0.0, 1.56], dtype=np.float32), radius=0.062)
        d_body = _smin(d_head, d_neck, k=0.05)

        d_clav_l = _sdf_capsule(P, np.array([0.0, 0.04, 1.41], dtype=np.float32), np.array([-0.16, 0.02, 1.41], dtype=np.float32), radius=0.015)
        d_clav_r = _sdf_capsule(P, np.array([0.0, 0.04, 1.41], dtype=np.float32), np.array([0.16, 0.02, 1.41], dtype=np.float32), radius=0.015)
        d_body = _smin(d_body, d_clav_l, k=0.025)
        d_body = _smin(d_body, d_clav_r, k=0.025)

        # 3. Torso + Pectoral Muscles + Six-Pack Abs
        chest_w = 0.20 if archetype != CharacterArchetype.GHOST_ROBED else 0.17
        d_chest = _sdf_ellipsoid(P, np.array([0.0, -0.01, 1.28], dtype=np.float32), np.array([chest_w, 0.11, 0.16], dtype=np.float32))

        # Pecs (Chest muscle definition)
        d_pec_l = _sdf_ellipsoid(P, np.array([-0.09, 0.065, 1.29], dtype=np.float32), np.array([0.075, 0.045, 0.06], dtype=np.float32))
        d_pec_r = _sdf_ellipsoid(P, np.array([0.09, 0.065, 1.29], dtype=np.float32), np.array([0.075, 0.045, 0.06], dtype=np.float32))
        d_chest = _smin(d_chest, d_pec_l, k=0.035)
        d_chest = _smin(d_chest, d_pec_r, k=0.035)

        d_waist = _sdf_ellipsoid(P, np.array([0.0, 0.0, 1.06], dtype=np.float32), np.array([chest_w * 0.82, 0.10, 0.14], dtype=np.float32))
        if archetype != CharacterArchetype.GHOST_ROBED:
            for ab_z in [1.14, 1.07, 1.00]:
                d_ab_l = _sdf_ellipsoid(P, np.array([-0.042, 0.065, ab_z], dtype=np.float32), np.array([0.032, 0.025, 0.025], dtype=np.float32))
                d_ab_r = _sdf_ellipsoid(P, np.array([0.042, 0.065, ab_z], dtype=np.float32), np.array([0.032, 0.025, 0.025], dtype=np.float32))
                d_waist = _smin(d_waist, d_ab_l, k=0.025)
                d_waist = _smin(d_waist, d_ab_r, k=0.025)

        d_hips = _sdf_ellipsoid(P, np.array([0.0, 0.0, 0.88], dtype=np.float32), np.array([chest_w * 0.90, 0.115, 0.13], dtype=np.float32))
        d_torso = _smin(d_chest, d_waist, k=0.07)
        d_torso = _smin(d_torso, d_hips, k=0.07)
        d_body = _smin(d_body, d_torso, k=0.07)

        # 4. Arms in A-Pose (Deltoids, Biceps, Forearm, Hands)
        arm_angle = math.radians(35.0)
        arm_len = 0.52
        sh_x = chest_w * 0.95

        # Deltoid shoulder caps
        d_delt_l = _sdf_ellipsoid(P, np.array([-sh_x, 0.0, 1.37], dtype=np.float32), np.array([0.06, 0.06, 0.06], dtype=np.float32))
        d_delt_r = _sdf_ellipsoid(P, np.array([sh_x, 0.0, 1.37], dtype=np.float32), np.array([0.06, 0.06, 0.06], dtype=np.float32))
        d_body = _smin(d_body, d_delt_l, k=0.04)
        d_body = _smin(d_body, d_delt_r, k=0.04)

        arm_l_start = np.array([-sh_x, 0.0, 1.35], dtype=np.float32)
        elbow_l = np.array([-sh_x - math.sin(arm_angle)*arm_len*0.5, 0.0, 1.35 - math.cos(arm_angle)*arm_len*0.5], dtype=np.float32)
        arm_l_end = np.array([-sh_x - math.sin(arm_angle)*arm_len, 0.0, 1.35 - math.cos(arm_angle)*arm_len], dtype=np.float32)

        d_upper_l = _sdf_capsule(P, arm_l_start, elbow_l, radius=0.052)
        d_lower_l = _sdf_capsule(P, elbow_l, arm_l_end, radius=0.044)
        d_arm_l = _smin(d_upper_l, d_lower_l, k=0.03)

        arm_r_start = np.array([sh_x, 0.0, 1.35], dtype=np.float32)
        elbow_r = np.array([sh_x + math.sin(arm_angle)*arm_len*0.5, 0.0, 1.35 - math.cos(arm_angle)*arm_len*0.5], dtype=np.float32)
        arm_r_end = np.array([sh_x + math.sin(arm_angle)*arm_len, 0.0, 1.35 - math.cos(arm_angle)*arm_len], dtype=np.float32)

        d_upper_r = _sdf_capsule(P, arm_r_start, elbow_r, radius=0.052)
        d_lower_r = _sdf_capsule(P, elbow_r, arm_r_end, radius=0.044)
        d_arm_r = _smin(d_upper_r, d_lower_r, k=0.03)

        # Hands with distinct thumb
        d_hand_l = _sdf_ellipsoid(P, arm_l_end, np.array([0.04, 0.022, 0.05], dtype=np.float32))
        d_thumb_l = _sdf_capsule(P, arm_l_end, arm_l_end + np.array([0.02, 0.02, -0.015], dtype=np.float32), radius=0.012)
        d_hand_l = _smin(d_hand_l, d_thumb_l, k=0.015)

        d_hand_r = _sdf_ellipsoid(P, arm_r_end, np.array([0.04, 0.022, 0.05], dtype=np.float32))
        d_thumb_r = _sdf_capsule(P, arm_r_end, arm_r_end + np.array([-0.02, 0.02, -0.015], dtype=np.float32), radius=0.012)
        d_hand_r = _smin(d_hand_r, d_thumb_r, k=0.015)

        d_arm_l = _smin(d_arm_l, d_hand_l, k=0.03)
        d_arm_r = _smin(d_arm_r, d_hand_r, k=0.03)

        d_body = _smin(d_body, d_arm_l, k=0.06)
        d_body = _smin(d_body, d_arm_r, k=0.06)

        # 5. Lower Body (Legs, Knee Caps, Calves, Boots OR Robe)
        if archetype == CharacterArchetype.GHOST_ROBED:
            gown_rad = 0.18 + np.maximum(0.88 - Z, 0.0) * 0.28
            d_gown = np.sqrt(X**2 + (Y * 1.1)**2) - gown_rad
            d_gown = np.maximum(d_gown, Z - 0.90)
            d_body = _smin(d_body, d_gown, k=0.08)

            d_hair_back = _sdf_capsule(P, np.array([0.0, -0.06, 1.68], dtype=np.float32), np.array([0.0, -0.10, 1.05], dtype=np.float32), radius=0.09)
            d_body = _smin(d_body, d_hair_back, k=0.04)
        else:
            leg_x = 0.11
            knee_z = 0.46
            knee_l = np.array([-leg_x, 0.0, knee_z], dtype=np.float32)
            knee_r = np.array([leg_x, 0.0, knee_z], dtype=np.float32)
            ankle_l = np.array([-leg_x, 0.0, 0.08], dtype=np.float32)
            ankle_r = np.array([leg_x, 0.0, 0.08], dtype=np.float32)

            d_thigh_l = _sdf_capsule(P, np.array([-leg_x, 0.0, 0.86], dtype=np.float32), knee_l, radius=0.075)
            d_thigh_r = _sdf_capsule(P, np.array([leg_x, 0.0, 0.86], dtype=np.float32), knee_r, radius=0.075)

            d_calf_l = _sdf_capsule(P, knee_l, ankle_l, radius=0.062)
            d_calf_r = _sdf_capsule(P, knee_r, ankle_r, radius=0.062)

            d_patella_l = _sdf_sphere(P, np.array([-leg_x, 0.055, knee_z], dtype=np.float32), radius=0.028)
            d_patella_r = _sdf_sphere(P, np.array([leg_x, 0.055, knee_z], dtype=np.float32), radius=0.028)

            d_leg_l = _smin(_smin(d_thigh_l, d_calf_l, k=0.035), d_patella_l, k=0.02)
            d_leg_r = _smin(_smin(d_thigh_r, d_calf_r, k=0.035), d_patella_r, k=0.02)

            d_foot_l = _sdf_box(P, np.array([-leg_x, 0.04, 0.04], dtype=np.float32), np.array([0.09, 0.20, 0.08], dtype=np.float32))
            d_foot_r = _sdf_box(P, np.array([leg_x, 0.04, 0.04], dtype=np.float32), np.array([0.09, 0.20, 0.08], dtype=np.float32))

            d_leg_l = _smin(d_leg_l, d_foot_l, k=0.03)
            d_leg_r = _smin(d_leg_r, d_foot_r, k=0.03)

            d_body = _smin(d_body, d_leg_l, k=0.07)
            d_body = _smin(d_body, d_leg_r, k=0.07)

        # Hair volume for all archetypes
        d_scalp = _sdf_ellipsoid(P, np.array([0.0, -0.02, 1.66], dtype=np.float32), np.array([0.13, 0.14, 0.13], dtype=np.float32))
        d_body = _smin(d_body, d_scalp, k=0.03)

        # 6. Extract Seamless Watertight Surface via Marching Cubes
        spacing = (vx[1] - vx[0], vy[1] - vy[0], vz[1] - vz[0])
        verts, faces, normals, _ = measure.marching_cubes(d_body, level=0.0, spacing=spacing)
        verts[:, 0] += vx[0]
        verts[:, 1] += vy[0]
        verts[:, 2] += vz[0]

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.remove_unreferenced_vertices()

        # Keep only the main connected component if any stray voxels exist
        split_parts = mesh.split(only_watertight=False)
        if len(split_parts) > 1:
            mesh = max(split_parts, key=lambda p: len(p.vertices))

        # 7. Anatomical PBR Vertex Coloring
        vx_c = mesh.vertices[:, 0]
        vy_c = mesh.vertices[:, 1]
        vz_c = mesh.vertices[:, 2]

        v_colors = np.full((len(mesh.vertices), 4), c_clothes, dtype=np.uint8)

        # Legs & Shoes
        is_leg = (vz_c < 0.85)
        v_colors[is_leg] = c_pants
        is_shoe = is_leg & (vz_c < 0.12)
        v_colors[is_shoe] = c_shoes

        # Tactical Belt
        if archetype != CharacterArchetype.GHOST_ROBED:
            is_belt = (vz_c >= 0.99) & (vz_c <= 1.05)
            v_colors[is_belt] = [18, 20, 26, 255]

        # Hands (extremities of A-pose arms)
        is_hand = (vz_c < 1.02) & (np.abs(vx_c) > 0.36)
        v_colors[is_hand] = c_skin

        # Neck
        is_neck = (vz_c >= 1.40) & (vz_c < 1.48) & (np.abs(vx_c) < 0.12)
        v_colors[is_neck] = c_skin

        # Head / Face
        is_head = (vz_c >= 1.48)
        # Hair: top, rear, and sides
        is_hair = is_head & ((vy_c < -0.015) | (vz_c >= 1.70) | (np.abs(vx_c) > 0.11))
        v_colors[is_hair] = c_hair

        # Face skin
        is_face = is_head & ~is_hair
        v_colors[is_face] = c_skin

        # Eyebrows
        is_eyebrow = is_face & (vy_c >= 0.075) & (vz_c >= 1.645) & (vz_c <= 1.668) & (np.abs(vx_c) >= 0.02) & (np.abs(vx_c) <= 0.07)
        v_colors[is_eyebrow] = c_hair

        # Eyes: left and right
        is_eye = is_face & (vy_c >= 0.08) & (np.abs(vz_c - 1.625) < 0.025) & (np.abs(vx_c) >= 0.025) & (np.abs(vx_c) <= 0.065)
        v_colors[is_eye] = c_eye_iris

        # Lips
        is_lip = is_face & (vy_c >= 0.08) & (np.abs(vz_c - 1.54) < 0.018) & (np.abs(vx_c) <= 0.035)
        v_colors[is_lip] = c_lips

        mesh.visual.vertex_colors = v_colors

        # 8. Transform from Z-up to Three.js standard Y-up
        rot_z_to_y = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        mesh.apply_transform(rot_z_to_y)

        # Ground base of mesh to Y=0
        mesh.apply_translation([0.0, -float(mesh.bounds[0][1]), 0.0])
        mesh.fix_normals()
        return mesh

    def generate_weapon_mesh(
        self,
        prompt: str,
        concept: Optional[OrthographicConceptPrompt] = None
    ) -> trimesh.Trimesh:
        archetype = self.detect_weapon_archetype(prompt)
        parts: List[trimesh.Trimesh] = []

        # Color definition for weapon parts
        c_blade = [0, 240, 255, 255]       # Glowing cyan plasma edge
        c_steel = [215, 225, 235, 255]     # Silver metallic blade spine
        c_gold = [225, 175, 45, 255]       # Antique gold / brass guard & pommel
        c_grip = [28, 28, 34, 255]         # Deep charcoal braided wrap
        c_body = [40, 44, 52, 255]         # Dark matte gunmetal
        c_scope = [0, 255, 180, 255]       # Glowing optic lens

        if archetype == WeaponArchetype.SWORD_BLADE:
            # Blade (Steel core)
            blade_core = trimesh.creation.box(extents=[0.014, 0.82, 0.035])
            blade_core.apply_translation([0.0, 0.41, 0.0])
            parts.append(_paint(blade_core, c_steel))

            # Blade Cutting Edge (Glowing Cyan)
            blade_edge = trimesh.creation.box(extents=[0.008, 0.82, 0.015])
            blade_edge.apply_translation([0.0, 0.41, 0.022])
            parts.append(_paint(blade_edge, c_blade))

            # Tsuba (Gold oval guard disk)
            tsuba = trimesh.creation.cylinder(radius=0.058, height=0.012)
            rot_tsuba = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
            tsuba.apply_transform(rot_tsuba)
            tsuba.apply_translation([0.0, 0.0, 0.0])
            parts.append(_paint(tsuba, c_gold))

            # Tsuka (Braided handle grip)
            handle = trimesh.creation.cylinder(radius=0.022, height=0.26)
            rot_handle = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
            handle.apply_transform(rot_handle)
            handle.apply_translation([0.0, -0.13, 0.0])
            parts.append(_paint(handle, c_grip))

            # Pommel end cap (Gold)
            pommel = trimesh.creation.icosphere(subdivisions=2, radius=0.026)
            pommel.apply_translation([0.0, -0.27, 0.0])
            parts.append(_paint(pommel, c_gold))

        elif archetype == WeaponArchetype.HAMMER_BLUNT:
            # Shaft (Dark grip)
            shaft = trimesh.creation.cylinder(radius=0.025, height=0.85)
            rot_shaft = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
            shaft.apply_transform(rot_shaft)
            shaft.apply_translation([0.0, 0.10, 0.0])
            parts.append(_paint(shaft, c_grip))

            # Heavy Hammer Head (Steel)
            head = trimesh.creation.box(extents=[0.14, 0.22, 0.16])
            head.apply_translation([0.0, 0.48, 0.0])
            parts.append(_paint(head, c_steel))

            # Gold trim plates
            trim = trimesh.creation.box(extents=[0.15, 0.06, 0.17])
            trim.apply_translation([0.0, 0.48, 0.0])
            parts.append(_paint(trim, c_gold))

        else:
            # Rifle / Gun / Blaster
            rot_b = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])

            # Barrel (Steel)
            barrel = trimesh.creation.cylinder(radius=0.018, height=0.62)
            barrel.apply_transform(rot_b)
            barrel.apply_translation([0.0, 0.42, 0.06])
            parts.append(_paint(barrel, c_steel))

            # Upper & Lower Receiver (Gunmetal)
            receiver = trimesh.creation.box(extents=[0.065, 0.38, 0.10])
            receiver.apply_translation([0.0, 0.06, 0.06])
            parts.append(_paint(receiver, c_body))

            # Handguard / Shroud (Dark)
            handguard = trimesh.creation.box(extents=[0.055, 0.32, 0.08])
            handguard.apply_translation([0.0, 0.30, 0.06])
            parts.append(_paint(handguard, c_body))

            # Pistol Grip (Charcoal)
            grip = trimesh.creation.box(extents=[0.04, 0.06, 0.15])
            rot_g = trimesh.transformations.rotation_matrix(math.radians(18), [1, 0, 0])
            grip.apply_transform(rot_g)
            grip.apply_translation([0.0, -0.06, -0.06])
            parts.append(_paint(grip, c_grip))

            # Magazine (Black)
            mag = trimesh.creation.box(extents=[0.035, 0.075, 0.16])
            mag.apply_translation([0.0, 0.08, -0.07])
            parts.append(_paint(mag, c_grip))

            # Buttstock (Gunmetal)
            stock = trimesh.creation.box(extents=[0.048, 0.24, 0.12])
            stock.apply_translation([0.0, -0.24, 0.04])
            parts.append(_paint(stock, c_body))

            # Optic Scope Tube
            scope = trimesh.creation.cylinder(radius=0.022, height=0.22)
            scope.apply_transform(rot_b)
            scope.apply_translation([0.0, 0.08, 0.14])
            parts.append(_paint(scope, c_steel))

            # Glowing optic lens
            lens = trimesh.creation.cylinder(radius=0.021, height=0.015)
            lens.apply_transform(rot_b)
            lens.apply_translation([0.0, 0.19, 0.14])
            parts.append(_paint(lens, c_scope))

        combined = trimesh.util.concatenate(parts)
        combined.apply_translation([0.0, -float(combined.bounds[0][1]), 0.0])
        combined.fix_normals()
        return combined
