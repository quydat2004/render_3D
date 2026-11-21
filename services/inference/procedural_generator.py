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
    ATHLETIC_RUNNER = "athletic_runner"
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
        if any(w in p for w in ["ma nữ", "bóng ma", "hồn ma", "ghost", "witch", "spirit", "robed", "áo dài", "dress", "kinh dị", "horror", "zombie"]):
            return CharacterArchetype.GHOST_ROBED
        elif any(w in p for w in ["chibi", "runner", "running", "chạy", "thể thao", "sport", "jersey", "athlete", "casual", "sneaker"]):
            return CharacterArchetype.ATHLETIC_RUNNER
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
        clothing_type = getattr(palette, "clothing_type", "robe" if archetype == CharacterArchetype.GHOST_ROBED else "suit")
        metalness = float(getattr(palette, "metalness", 0.05 if archetype != CharacterArchetype.MECHA_HEAVY else 0.85))
        roughness = float(getattr(palette, "roughness", 0.82 if archetype != CharacterArchetype.MECHA_HEAVY else 0.25))

        tex_set = tex_engine.generate_character_textures(archetype.value, width=1024, height=1024, palette=palette)

        skin_rgb = _hex_to_rgb(getattr(palette, "skin_color_hex", "#ECCDBA"))[:3]
        hair_rgb = _hex_to_rgb(getattr(palette, "hair_color_hex", "#2D1F17"))[:3]
        default_cloth_hex = "#EAB308" if archetype == CharacterArchetype.ATHLETIC_RUNNER else ("#F5F8FC" if archetype == CharacterArchetype.GHOST_ROBED else "#2563EB")
        cloth_rgb = _hex_to_rgb(getattr(palette, "primary_cloth_hex", default_cloth_hex))[:3]

        # 1. Base Body with Dedicated Skin PBR Material (Smooth natural skin + Zone MetallicRoughness)
        body_mesh = generate_bony_base_body(archetype.value)
        pbr_body = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[c / 255.0 for c in skin_rgb] + [1.0],
            baseColorTexture=tex_set.albedo,
            normalTexture=tex_set.normal,
            metallicRoughnessTexture=tex_set.metallic_roughness,
            metallicFactor=0.0,
            roughnessFactor=0.55
        )
        body_mesh.visual = trimesh.visual.TextureVisuals(uv=body_mesh.visual.uv, material=pbr_body)

        # 2. Outfit Geometry: Solid Plate Armor OR Flowing Fabric Shell
        if clothing_type in ["armor", "knight", "mecha"] or metalness > 0.6:
            outfit_mesh = generate_plate_armor(archetype.value)
            pbr_outfit = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[c / 255.0 for c in cloth_rgb] + [1.0],
                normalTexture=tex_set.normal,
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
                metallicRoughnessTexture=tex_set.metallic_roughness,
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
        elif archetype == CharacterArchetype.ATHLETIC_RUNNER:
            c_skin = [238, 198, 172, 255]     # Warm healthy skin
            c_hair = [45, 32, 26, 255]        # Brown styled hair
            c_eye_iris = [35, 30, 28, 255]    # Natural iris
            c_lips = [190, 85, 90, 255]       # Soft lips
            c_clothes = [234, 179, 8, 255]    # Athletic yellow jersey
            c_pants = [24, 30, 42, 255]       # Dark sports shorts / tights
            c_shoes = [245, 248, 252, 255]    # Clean white running sneakers
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
        p_lower = prompt.lower()

        # Dynamic energy/accent palette from prompt keywords
        if any(w in p_lower for w in ["đỏ", "red", "crimson", "huyết", "fire", "lửa", "blood"]):
            c_blade = [255, 45, 55, 255]
            c_scope = [255, 60, 40, 255]
        elif any(w in p_lower for w in ["vàng", "gold", "golden", "hoàng kim", "solar", "ánh sáng"]):
            c_blade = [255, 215, 55, 255]
            c_scope = [255, 230, 90, 255]
        elif any(w in p_lower for w in ["xanh lá", "green", "emerald", "jade", "ngọc", "acid", "독"]):
            c_blade = [35, 245, 115, 255]
            c_scope = [50, 255, 140, 255]
        elif any(w in p_lower for w in ["tím", "purple", "violet", "void", "hắc ám", "shadow"]):
            c_blade = [185, 65, 255, 255]
            c_scope = [210, 95, 255, 255]
        else:
            c_blade = [0, 240, 255, 255]       # Glowing cyan plasma edge
            c_scope = [0, 255, 180, 255]       # Glowing optic lens

        c_steel = [215, 225, 235, 255]         # High-carbon silver steel
        c_dark_steel = [120, 130, 145, 255]    # Recessed fuller / barrel flutes
        c_gold = [225, 175, 45, 255]           # Antique gold / brass guard & trim
        c_grip = [28, 28, 34, 255]             # Deep charcoal braided wrap / polymer
        c_body = [44, 48, 58, 255]             # Dark matte gunmetal receiver
        c_crimson_wrap = [135, 24, 32, 255]    # Traditional rayskin under-wrap

        rot_y_cyl = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])

        if archetype == WeaponArchetype.SWORD_BLADE:
            # 1. Segmented Curved Blade with Tapered Kissaki Tip, Beveled Edge & Fuller Groove
            num_segs = 10
            blade_len = 0.86
            seg_len = blade_len / num_segs
            for i in range(num_segs):
                t0 = i / float(num_segs)
                t1 = (i + 1) / float(num_segs)
                t_mid = 0.5 * (t0 + t1)
                y_pos = 0.03 + (i + 0.5) * seg_len
                # Subtle katana sori curvature along +Z
                curve_z = (t_mid ** 1.8) * 0.024
                # Taper width & depth near kissaki tip
                taper = 1.0 - max(0.0, (t_mid - 0.72) / 0.28) * 0.65

                # Spine (Mune - Polished Steel)
                spine = trimesh.creation.box(extents=[0.013 * taper, seg_len * 1.04, 0.024 * taper])
                spine.apply_translation([0.0, y_pos, curve_z - 0.005 * taper])
                parts.append(_paint(spine, c_steel))

                # Fuller Blood Groove (Bo-Hi - Dark Recessed Steel along first 75% of blade)
                if t_mid < 0.78:
                    fuller = trimesh.creation.box(extents=[0.015 * taper, seg_len * 0.96, 0.006 * taper])
                    fuller.apply_translation([0.0, y_pos, curve_z - 0.008 * taper])
                    parts.append(_paint(fuller, c_dark_steel))

                # Beveled Cutting Edge (Ha - Glowing Plasma / Tempered Hamon Edge)
                edge = trimesh.creation.box(extents=[0.006 * taper, seg_len * 1.04, 0.014 * taper])
                edge.apply_translation([0.0, y_pos, curve_z + 0.013 * taper])
                parts.append(_paint(edge, c_blade))

            # 2. Habaki (Brass/Gold Blade Locking Collar) & Seppa Spacers
            habaki = trimesh.creation.box(extents=[0.020, 0.034, 0.042])
            habaki.apply_translation([0.0, 0.022, 0.002])
            parts.append(_paint(habaki, c_gold))

            seppa_f = trimesh.creation.cylinder(radius=0.026, height=0.005, sections=20)
            seppa_f.apply_transform(rot_y_cyl)
            seppa_f.apply_translation([0.0, 0.006, 0.0])
            parts.append(_paint(seppa_f, c_gold))

            # 3. Ornate Tsuba (Handguard with Outer Gold Rim, Dark Core & 4 Bosses)
            tsuba_core = trimesh.creation.cylinder(radius=0.056, height=0.010, sections=28)
            tsuba_core.apply_transform(rot_y_cyl)
            parts.append(_paint(tsuba_core, c_body))

            tsuba_rim = trimesh.creation.cylinder(radius=0.062, height=0.007, sections=28)
            tsuba_rim.apply_transform(rot_y_cyl)
            parts.append(_paint(tsuba_rim, c_gold))

            for angle_deg in [0, 90, 180, 270]:
                rad = math.radians(angle_deg)
                boss = trimesh.creation.cylinder(radius=0.009, height=0.014, sections=12)
                boss.apply_transform(rot_y_cyl)
                boss.apply_translation([math.cos(rad) * 0.038, 0.0, math.sin(rad) * 0.038])
                parts.append(_paint(boss, c_gold))

            # 4. Tsuka Handle Core + Alternating Tsuka-Ito Braid Bands + Menuki Ornaments
            tsuka_core = trimesh.creation.cylinder(radius=0.018, height=0.26, sections=20)
            tsuka_core.apply_transform(rot_y_cyl)
            tsuka_core.vertices[:, 0] *= 0.78  # Oval ergonomic cross-section
            tsuka_core.apply_translation([0.0, -0.135, 0.0])
            parts.append(_paint(tsuka_core, c_crimson_wrap))

            for band_idx in range(9):
                by = -0.022 - band_idx * 0.027
                band = trimesh.creation.cylinder(radius=0.021, height=0.015, sections=20)
                band.apply_transform(rot_y_cyl)
                band.vertices[:, 0] *= 0.80
                rot_tilt = trimesh.transformations.rotation_matrix(
                    math.radians(14 if band_idx % 2 == 0 else -14), [0, 0, 1]
                )
                band.apply_transform(rot_tilt)
                band.apply_translation([0.0, by, 0.0])
                parts.append(_paint(band, c_grip))

            for sign in [-1, 1]:
                menuki = trimesh.creation.icosphere(subdivisions=1, radius=0.008)
                menuki.apply_translation([sign * 0.016, -0.11, 0.0])
                parts.append(_paint(menuki, c_gold))

            # 5. Fuchi & Kashira (Ferrule + Pommel Cap + Glowing Energy Ring)
            fuchi = trimesh.creation.cylinder(radius=0.022, height=0.016, sections=20)
            fuchi.apply_transform(rot_y_cyl)
            fuchi.vertices[:, 0] *= 0.82
            fuchi.apply_translation([0.0, -0.010, 0.0])
            parts.append(_paint(fuchi, c_gold))

            kashira = trimesh.creation.cylinder(radius=0.022, height=0.020, sections=20)
            kashira.apply_transform(rot_y_cyl)
            kashira.vertices[:, 0] *= 0.82
            kashira.apply_translation([0.0, -0.270, 0.0])
            parts.append(_paint(kashira, c_gold))

            pommel_gem = trimesh.creation.icosphere(subdivisions=2, radius=0.015)
            pommel_gem.apply_translation([0.0, -0.285, 0.0])
            parts.append(_paint(pommel_gem, c_blade))

        elif archetype == WeaponArchetype.HAMMER_BLUNT:
            # 1. Reinforced Warhammer Haft with Ribbed Grip & Flanged Pommel
            shaft = trimesh.creation.cylinder(radius=0.022, height=0.88, sections=24)
            shaft.apply_transform(rot_y_cyl)
            shaft.apply_translation([0.0, 0.10, 0.0])
            parts.append(_paint(shaft, c_dark_steel))

            for g_idx in range(7):
                gy = -0.30 + g_idx * 0.045
                grip_ring = trimesh.creation.cylinder(radius=0.026, height=0.028, sections=20)
                grip_ring.apply_transform(rot_y_cyl)
                grip_ring.apply_translation([0.0, gy, 0.0])
                parts.append(_paint(grip_ring, c_grip))

            pommel_flange = trimesh.creation.cylinder(radius=0.036, height=0.025, sections=16)
            pommel_flange.apply_transform(rot_y_cyl)
            pommel_flange.apply_translation([0.0, -0.34, 0.0])
            parts.append(_paint(pommel_flange, c_gold))

            # 2. Central Reinforced Collar & Top Spear Spike
            collar = trimesh.creation.box(extents=[0.075, 0.24, 0.075])
            collar.apply_translation([0.0, 0.46, 0.0])
            parts.append(_paint(collar, c_gold))

            top_spike = trimesh.creation.cone(radius=0.026, height=0.11, sections=12)
            rot_spike_y = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
            top_spike.apply_transform(rot_spike_y)
            top_spike.apply_translation([0.0, 0.56, 0.0])
            parts.append(_paint(top_spike, c_steel))

            # 3. Heavy Hammer Core Block + Beveled Striking Face + Kinetic Plasma Core
            head_core = trimesh.creation.box(extents=[0.13, 0.18, 0.18])
            head_core.apply_translation([0.0, 0.47, 0.01])
            parts.append(_paint(head_core, c_body))

            strike_neck = trimesh.creation.cylinder(radius=0.068, height=0.06, sections=8)
            strike_neck.apply_translation([0.0, 0.47, 0.11])
            parts.append(_paint(strike_neck, c_steel))

            strike_face = trimesh.creation.cylinder(radius=0.082, height=0.035, sections=8)
            strike_face.apply_translation([0.0, 0.47, 0.145])
            parts.append(_paint(strike_face, c_steel))

            energy_rings = trimesh.creation.cylinder(radius=0.074, height=0.016, sections=16)
            energy_rings.apply_translation([0.0, 0.47, 0.125])
            parts.append(_paint(energy_rings, c_blade))

            # 4. Rear Armor-Piercing War-Pick Spike
            back_spike = trimesh.creation.cone(radius=0.048, height=0.16, sections=8)
            rot_back = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
            back_spike.apply_transform(rot_back)
            back_spike.apply_translation([0.0, 0.47, -0.08])
            parts.append(_paint(back_spike, c_steel))

            # 5. Gold Cheek Trim Plates & Rivet Studs
            for sign in [-1, 1]:
                cheek = trimesh.creation.box(extents=[0.012, 0.14, 0.14])
                cheek.apply_translation([sign * 0.068, 0.47, 0.01])
                parts.append(_paint(cheek, c_gold))
                for ry in [0.42, 0.52]:
                    for rz in [-0.03, 0.05]:
                        rivet = trimesh.creation.icosphere(subdivisions=1, radius=0.008)
                        rivet.apply_translation([sign * 0.076, ry, rz])
                        parts.append(_paint(rivet, c_blade))

        else:
            # High-Detail Tactical / Sci-Fi Rifle
            # 1. Heavy Fluted Barrel & Parallel Gas Block Tube
            barrel = trimesh.creation.cylinder(radius=0.016, height=0.62, sections=24)
            barrel.apply_transform(rot_y_cyl)
            barrel.apply_translation([0.0, 0.43, 0.065])
            parts.append(_paint(barrel, c_steel))

            gas_tube = trimesh.creation.cylinder(radius=0.009, height=0.42, sections=16)
            gas_tube.apply_transform(rot_y_cyl)
            gas_tube.apply_translation([0.0, 0.34, 0.092])
            parts.append(_paint(gas_tube, c_dark_steel))

            # 2. Multi-Baffle Muzzle Brake & Glowing Plasma Bore
            muzzle_body = trimesh.creation.cylinder(radius=0.024, height=0.09, sections=16)
            muzzle_body.apply_transform(rot_y_cyl)
            muzzle_body.apply_translation([0.0, 0.74, 0.065])
            parts.append(_paint(muzzle_body, c_body))

            for b_i in range(3):
                baffle = trimesh.creation.cylinder(radius=0.027, height=0.012, sections=16)
                baffle.apply_transform(rot_y_cyl)
                baffle.apply_translation([0.0, 0.71 + b_i * 0.028, 0.065])
                parts.append(_paint(baffle, c_dark_steel))

            muzzle_glow = trimesh.creation.cylinder(radius=0.013, height=0.015, sections=16)
            muzzle_glow.apply_transform(rot_y_cyl)
            muzzle_glow.apply_translation([0.0, 0.785, 0.065])
            parts.append(_paint(muzzle_glow, c_blade))

            # 3. Ventilated Handguard Shroud + Glowing Plasma Conduits
            handguard = trimesh.creation.box(extents=[0.056, 0.34, 0.078])
            handguard.apply_translation([0.0, 0.33, 0.062])
            parts.append(_paint(handguard, c_body))

            for v_i in range(6):
                vy = 0.19 + v_i * 0.052
                vent = trimesh.creation.box(extents=[0.062, 0.022, 0.036])
                vent.apply_translation([0.0, vy, 0.065])
                parts.append(_paint(vent, c_dark_steel))

            for sign in [-1, 1]:
                conduit = trimesh.creation.box(extents=[0.006, 0.30, 0.010])
                conduit.apply_translation([sign * 0.029, 0.33, 0.065])
                parts.append(_paint(conduit, c_blade))

            # 4. Upper & Lower Receiver + Ejection Port + Side Charging Handle
            upper_rec = trimesh.creation.box(extents=[0.064, 0.36, 0.072])
            upper_rec.apply_translation([0.0, 0.03, 0.072])
            parts.append(_paint(upper_rec, c_body))

            lower_rec = trimesh.creation.box(extents=[0.058, 0.28, 0.055])
            lower_rec.apply_translation([0.0, 0.02, 0.022])
            parts.append(_paint(lower_rec, c_dark_steel))

            eject_port = trimesh.creation.box(extents=[0.068, 0.085, 0.024])
            eject_port.apply_translation([0.0, 0.07, 0.075])
            parts.append(_paint(eject_port, c_steel))

            charging_handle = trimesh.creation.box(extents=[0.045, 0.018, 0.012])
            charging_handle.apply_translation([-0.038, 0.14, 0.082])
            parts.append(_paint(charging_handle, c_gold))

            # 5. Full-Length Top Picatinny Rail (14 Machined Recoil Lugs)
            rail_base = trimesh.creation.box(extents=[0.032, 0.56, 0.010])
            rail_base.apply_translation([0.0, 0.18, 0.110])
            parts.append(_paint(rail_base, c_dark_steel))

            for r_i in range(14):
                ry = -0.07 + r_i * 0.038
                lug = trimesh.creation.box(extents=[0.036, 0.018, 0.008])
                lug.apply_translation([0.0, ry, 0.118])
                parts.append(_paint(lug, c_steel))

            # 6. Trigger Guard, Curved Trigger & Ergonomic Pistol Grip
            trig_guard = trimesh.creation.box(extents=[0.018, 0.075, 0.042])
            trig_guard.apply_translation([0.0, -0.005, -0.018])
            parts.append(_paint(trig_guard, c_body))

            trigger = trimesh.creation.cylinder(radius=0.005, height=0.032, sections=10)
            rot_trig = trimesh.transformations.rotation_matrix(math.radians(25), [1, 0, 0])
            trigger.apply_transform(rot_trig)
            trigger.apply_translation([0.0, -0.005, -0.012])
            parts.append(_paint(trigger, c_gold))

            grip = trimesh.creation.box(extents=[0.038, 0.058, 0.15])
            rot_g = trimesh.transformations.rotation_matrix(math.radians(20), [1, 0, 0])
            grip.apply_transform(rot_g)
            grip.apply_translation([0.0, -0.075, -0.065])
            parts.append(_paint(grip, c_grip))

            # 7. Flared Magwell & Ribbed Tactical Magazine
            magwell = trimesh.creation.box(extents=[0.052, 0.095, 0.040])
            magwell.apply_translation([0.0, 0.095, 0.002])
            parts.append(_paint(magwell, c_body))

            mag = trimesh.creation.box(extents=[0.036, 0.076, 0.17])
            rot_m = trimesh.transformations.rotation_matrix(math.radians(-10), [1, 0, 0])
            mag.apply_transform(rot_m)
            mag.apply_translation([0.0, 0.105, -0.085])
            parts.append(_paint(mag, c_grip))

            for m_i in range(4):
                mz = -0.035 - m_i * 0.034
                m_rib = trimesh.creation.box(extents=[0.040, 0.080, 0.010])
                m_rib.apply_transform(rot_m)
                m_rib.apply_translation([0.0, 0.105 + m_i * 0.005, mz])
                parts.append(_paint(m_rib, c_dark_steel))

            # 8. Adjustable Tactical Buttstock with Cheek Riser & Rubber Recoil Pad
            buffer_tube = trimesh.creation.cylinder(radius=0.019, height=0.24, sections=16)
            buffer_tube.apply_transform(rot_y_cyl)
            buffer_tube.apply_translation([0.0, -0.25, 0.068])
            parts.append(_paint(buffer_tube, c_dark_steel))

            stock_body = trimesh.creation.box(extents=[0.046, 0.22, 0.115])
            stock_body.apply_translation([0.0, -0.26, 0.042])
            parts.append(_paint(stock_body, c_body))

            cheek_rest = trimesh.creation.box(extents=[0.052, 0.15, 0.032])
            cheek_rest.apply_translation([0.0, -0.24, 0.095])
            parts.append(_paint(cheek_rest, c_grip))

            butt_pad = trimesh.creation.box(extents=[0.050, 0.025, 0.135])
            butt_pad.apply_translation([0.0, -0.375, 0.038])
            parts.append(_paint(butt_pad, c_grip))

            # 9. Dual-Ring Precision Optic Scope + Turrets + Glowing Objective & Ocular Lenses
            for mount_y in [-0.01, 0.11]:
                mount = trimesh.creation.box(extents=[0.044, 0.026, 0.045])
                mount.apply_translation([0.0, mount_y, 0.138])
                parts.append(_paint(mount, c_dark_steel))

            scope_tube = trimesh.creation.cylinder(radius=0.020, height=0.25, sections=24)
            scope_tube.apply_transform(rot_y_cyl)
            scope_tube.apply_translation([0.0, 0.05, 0.160])
            parts.append(_paint(scope_tube, c_body))

            obj_bell = trimesh.creation.cylinder(radius=0.028, height=0.065, sections=24)
            obj_bell.apply_transform(rot_y_cyl)
            obj_bell.apply_translation([0.0, 0.165, 0.160])
            parts.append(_paint(obj_bell, c_steel))

            ocular_bell = trimesh.creation.cylinder(radius=0.024, height=0.045, sections=24)
            ocular_bell.apply_transform(rot_y_cyl)
            ocular_bell.apply_translation([0.0, -0.075, 0.160])
            parts.append(_paint(ocular_bell, c_steel))

            # Scope Elevation Turret
            turret = trimesh.creation.cylinder(radius=0.012, height=0.022, sections=14)
            turret.apply_translation([0.0, 0.05, 0.185])
            parts.append(_paint(turret, c_gold))

            # Glowing Front & Rear Optic Lenses
            lens_f = trimesh.creation.cylinder(radius=0.025, height=0.010, sections=24)
            lens_f.apply_transform(rot_y_cyl)
            lens_f.apply_translation([0.0, 0.196, 0.160])
            parts.append(_paint(lens_f, c_scope))

            lens_r = trimesh.creation.cylinder(radius=0.021, height=0.010, sections=24)
            lens_r.apply_transform(rot_y_cyl)
            lens_r.apply_translation([0.0, -0.096, 0.160])
            parts.append(_paint(lens_r, c_scope))

            # 10. Underbarrel Tactical Laser Sight Module
            laser_box = trimesh.creation.box(extents=[0.034, 0.095, 0.028])
            laser_box.apply_translation([0.0, 0.42, 0.012])
            parts.append(_paint(laser_box, c_dark_steel))

            laser_emitter = trimesh.creation.cylinder(radius=0.008, height=0.015, sections=12)
            laser_emitter.apply_transform(rot_y_cyl)
            laser_emitter.apply_translation([0.0, 0.472, 0.012])
            parts.append(_paint(laser_emitter, c_blade))

        combined = trimesh.util.concatenate(parts)
        combined.apply_translation([0.0, -float(combined.bounds[0][1]), 0.0])
        combined.fix_normals()
        return combined
