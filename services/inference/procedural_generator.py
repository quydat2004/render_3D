import math
from enum import Enum
from typing import Optional, List
import numpy as np
import trimesh

from services.orchestrator.gemini_orchestrator import OrthographicConceptPrompt

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

    def generate_character_mesh(
        self,
        prompt: str,
        concept: Optional[OrthographicConceptPrompt] = None
    ) -> trimesh.Trimesh:
        archetype = self.detect_character_archetype(prompt)
        parts: List[trimesh.Trimesh] = []

        # Color definition
        if archetype == CharacterArchetype.GHOST_ROBED:
            c_skin = [235, 240, 245, 255]     # Pale ghostly white/cyan
            c_hair = [15, 16, 22, 255]        # Raven black hair
            c_eye_iris = [210, 20, 25, 255]   # Horror red glowing eyes
            c_lips = [140, 35, 45, 255]       # Dark blood crimson lips
            c_clothes = [245, 248, 252, 255]  # Burial white shroud/dress
        elif archetype == CharacterArchetype.MECHA_HEAVY:
            c_skin = [110, 120, 135, 255]     # Steel chassis
            c_hair = [40, 45, 55, 255]        # Carbon plating
            c_eye_iris = [0, 240, 255, 255]   # Glowing cyan optics
            c_lips = [30, 35, 40, 255]        # Grille
            c_clothes = [50, 60, 75, 255]     # Heavy armor plating
        else:
            c_skin = [230, 190, 165, 255]     # Natural human skin
            c_hair = [35, 28, 24, 255]        # Dark brown/black hair
            c_eye_iris = [30, 30, 35, 255]    # Natural dark iris
            c_lips = [185, 75, 80, 255]       # Rose lips
            c_clothes = [40, 48, 62, 255]     # Tactical navy/carbon suit

        c_eye_white = [250, 250, 255, 255]

        # 1. Head & Cranium
        head = trimesh.creation.icosphere(subdivisions=2, radius=0.12)
        head.apply_translation([0.0, 0.0, 1.62])
        parts.append(_paint(head, c_skin))

        # 2. Facial Features (Eyes, Nose, Mouth)
        # Left Eye (Sclera + Pupil)
        eye_l = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
        eye_l.apply_translation([-0.042, 0.11, 1.635])
        parts.append(_paint(eye_l, c_eye_white))

        pupil_l = trimesh.creation.icosphere(subdivisions=1, radius=0.012)
        pupil_l.apply_translation([-0.042, 0.124, 1.635])
        parts.append(_paint(pupil_l, c_eye_iris))

        # Right Eye (Sclera + Pupil)
        eye_r = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
        eye_r.apply_translation([0.042, 0.11, 1.635])
        parts.append(_paint(eye_r, c_eye_white))

        pupil_r = trimesh.creation.icosphere(subdivisions=1, radius=0.012)
        pupil_r.apply_translation([0.042, 0.124, 1.635])
        parts.append(_paint(pupil_r, c_eye_iris))

        # Nose bridge and tip
        nose = trimesh.creation.cone(radius=0.018, height=0.038)
        rot_nose = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        nose.apply_transform(rot_nose)
        nose.apply_translation([0.0, 0.122, 1.595])
        parts.append(_paint(nose, c_skin))

        # Mouth & Lips
        lips = trimesh.creation.box(extents=[0.048, 0.016, 0.014])
        lips.apply_translation([0.0, 0.116, 1.545])
        parts.append(_paint(lips, c_lips))

        # 3. Hair (Top scalp volume + long draping strands)
        scalp_hair = trimesh.creation.icosphere(subdivisions=2, radius=0.128)
        scalp_hair.apply_translation([0.0, -0.02, 1.65])
        parts.append(_paint(scalp_hair, c_hair))

        # Flowing back hair
        back_hair = trimesh.creation.box(extents=[0.24, 0.15, 0.85])
        back_hair.apply_translation([0.0, -0.07, 1.28])
        parts.append(_paint(back_hair, c_hair))

        # Side hair strands framing face
        side_hair_l = trimesh.creation.box(extents=[0.05, 0.12, 0.65])
        side_hair_l.apply_translation([-0.11, 0.02, 1.35])
        parts.append(_paint(side_hair_l, c_hair))

        side_hair_r = trimesh.creation.box(extents=[0.05, 0.12, 0.65])
        side_hair_r.apply_translation([0.11, 0.02, 1.35])
        parts.append(_paint(side_hair_r, c_hair))

        # 4. Neck
        neck = trimesh.creation.cylinder(radius=0.06, height=0.14)
        neck.apply_translation([0.0, 0.0, 1.48])
        parts.append(_paint(neck, c_skin))

        # 5. Torso (Chest & Waist)
        chest_width = 0.38 if archetype != CharacterArchetype.GHOST_ROBED else 0.32
        chest = trimesh.creation.box(extents=[chest_width, 0.22, 0.36])
        chest.apply_translation([0.0, 0.0, 1.25])
        parts.append(_paint(chest, c_clothes))

        waist = trimesh.creation.box(extents=[chest_width * 0.85, 0.18, 0.28])
        waist.apply_translation([0.0, 0.0, 0.98])
        parts.append(_paint(waist, c_clothes))

        # 6. Arms in A-Pose (Angled at 40 degrees)
        arm_angle = math.radians(40.0)
        arm_len = 0.58
        arm_r = 0.055

        # Left Arm
        arm_l = trimesh.creation.cylinder(radius=arm_r, height=arm_len)
        rot_l = trimesh.transformations.rotation_matrix(arm_angle, [0, 1, 0])
        arm_l.apply_transform(rot_l)
        arm_l.apply_translation([
            -(chest_width * 0.5 + math.sin(arm_angle) * (arm_len * 0.5)),
            0.0,
            1.25 - math.cos(arm_angle) * (arm_len * 0.5)
        ])
        parts.append(_paint(arm_l, c_clothes))

        # Right Arm
        arm_r_mesh = trimesh.creation.cylinder(radius=arm_r, height=arm_len)
        rot_r = trimesh.transformations.rotation_matrix(-arm_angle, [0, 1, 0])
        arm_r_mesh.apply_transform(rot_r)
        arm_r_mesh.apply_translation([
            chest_width * 0.5 + math.sin(arm_angle) * (arm_len * 0.5),
            0.0,
            1.25 - math.cos(arm_angle) * (arm_len * 0.5)
        ])
        parts.append(_paint(arm_r_mesh, c_clothes))

        # Hands
        hand_l = trimesh.creation.icosphere(subdivisions=1, radius=0.045)
        hand_l.apply_translation([
            -(chest_width * 0.5 + math.sin(arm_angle) * arm_len),
            0.0,
            1.25 - math.cos(arm_angle) * arm_len
        ])
        parts.append(_paint(hand_l, c_skin))

        hand_r = trimesh.creation.icosphere(subdivisions=1, radius=0.045)
        hand_r.apply_translation([
            chest_width * 0.5 + math.sin(arm_angle) * arm_len,
            0.0,
            1.25 - math.cos(arm_angle) * arm_len
        ])
        parts.append(_paint(hand_r, c_skin))

        # 7. Lower Body: Flowing Robe/Gown for Ghost, or Legs for Warrior
        if archetype == CharacterArchetype.GHOST_ROBED:
            gown = trimesh.creation.cylinder(radius=0.28, height=0.88, sections=24)
            scale_factor = (1.0 + (1.0 - gown.vertices[:, 2] / 0.88) * 0.6)[:, np.newaxis]
            gown.vertices[:, :2] *= scale_factor
            gown.apply_translation([0.0, 0.0, 0.44])
            parts.append(_paint(gown, c_clothes))
        else:
            leg_spacing = 0.12
            c_pants = [30, 36, 48, 255]
            c_shoes = [20, 22, 26, 255]
            for sign in [-1, 1]:
                thigh = trimesh.creation.cylinder(radius=0.08, height=0.44)
                thigh.apply_translation([sign * leg_spacing, 0.0, 0.62])
                parts.append(_paint(thigh, c_pants))

                calf = trimesh.creation.cylinder(radius=0.065, height=0.42)
                calf.apply_translation([sign * leg_spacing, 0.0, 0.22])
                parts.append(_paint(calf, c_pants))

                foot = trimesh.creation.box(extents=[0.09, 0.22, 0.08])
                foot.apply_translation([sign * leg_spacing, 0.04, 0.04])
                parts.append(_paint(foot, c_shoes))

        combined = trimesh.util.concatenate(parts)
        # Convert from CAD/Blender Z-up to glTF/Three.js standard Y-up
        rot_z_to_y = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        combined.apply_transform(rot_z_to_y)
        # Ground base of mesh to Y=0
        combined.apply_translation([0.0, -float(combined.bounds[0][1]), 0.0])
        combined.fix_normals()
        return combined

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
