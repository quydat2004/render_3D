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

class Procedural3DGenerator:
    def __init__(self):
        pass

    def detect_character_archetype(self, prompt: str) -> CharacterArchetype:
        p = prompt.lower()
        if any(w in p for w in ["ma", "nữ", "ghost", "witch", "spirit", "robed", "áo dài", "dress", "gặp ma", "kinh dị"]):
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

        # 1. Head & Face
        head = trimesh.creation.icosphere(subdivisions=2, radius=0.12)
        head.apply_translation([0.0, 0.0, 1.62])
        parts.append(head)

        # Hair (if ghost / long hair)
        if archetype == CharacterArchetype.GHOST_ROBED:
            hair = trimesh.creation.box(extents=[0.25, 0.16, 0.85])
            hair.apply_translation([0.0, -0.06, 1.30])
            parts.append(hair)

        # 2. Neck
        neck = trimesh.creation.cylinder(radius=0.06, height=0.14)
        neck.apply_translation([0.0, 0.0, 1.48])
        parts.append(neck)

        # 3. Torso (Chest & Waist)
        chest_width = 0.38 if archetype != CharacterArchetype.GHOST_ROBED else 0.32
        chest = trimesh.creation.box(extents=[chest_width, 0.22, 0.36])
        chest.apply_translation([0.0, 0.0, 1.25])
        parts.append(chest)

        waist = trimesh.creation.box(extents=[chest_width * 0.85, 0.18, 0.28])
        waist.apply_translation([0.0, 0.0, 0.98])
        parts.append(waist)

        # 4. Arms in A-Pose (Angled at 40 degrees)
        arm_angle = math.radians(40.0)
        arm_len = 0.58
        arm_r = 0.055

        # Left Arm (Extends in -X direction)
        arm_l = trimesh.creation.cylinder(radius=arm_r, height=arm_len)
        rot_l = trimesh.transformations.rotation_matrix(arm_angle, [0, 1, 0])
        arm_l.apply_transform(rot_l)
        arm_l.apply_translation([
            -(chest_width * 0.5 + math.sin(arm_angle) * (arm_len * 0.5)),
            0.0,
            1.25 - math.cos(arm_angle) * (arm_len * 0.5)
        ])
        parts.append(arm_l)

        # Right Arm (Extends in +X direction)
        arm_r_mesh = trimesh.creation.cylinder(radius=arm_r, height=arm_len)
        rot_r = trimesh.transformations.rotation_matrix(-arm_angle, [0, 1, 0])
        arm_r_mesh.apply_transform(rot_r)
        arm_r_mesh.apply_translation([
            chest_width * 0.5 + math.sin(arm_angle) * (arm_len * 0.5),
            0.0,
            1.25 - math.cos(arm_angle) * (arm_len * 0.5)
        ])
        parts.append(arm_r_mesh)

        # 5. Lower Body: Flowing Robe/Gown for Ghost, or Legs for Warrior
        if archetype == CharacterArchetype.GHOST_ROBED:
            # Flowing Asian ghost robe / áo dài from waist down to ground
            gown = trimesh.creation.cylinder(radius=0.28, height=0.88, sections=24)
            scale_factor = (1.0 + (1.0 - gown.vertices[:, 2] / 0.88) * 0.6)[:, np.newaxis]
            gown.vertices[:, :2] *= scale_factor
            gown.apply_translation([0.0, 0.0, 0.44])
            parts.append(gown)
        else:
            # Humanoid Legs
            leg_spacing = 0.12
            for sign in [-1, 1]:
                thigh = trimesh.creation.cylinder(radius=0.08, height=0.44)
                thigh.apply_translation([sign * leg_spacing, 0.0, 0.62])
                parts.append(thigh)

                calf = trimesh.creation.cylinder(radius=0.065, height=0.42)
                calf.apply_translation([sign * leg_spacing, 0.0, 0.22])
                parts.append(calf)

                foot = trimesh.creation.box(extents=[0.09, 0.22, 0.08])
                foot.apply_translation([sign * leg_spacing, 0.04, 0.04])
                parts.append(foot)

        # Merge all body parts
        combined = trimesh.util.concatenate(parts)
        combined.fix_normals()
        return combined

    def generate_weapon_mesh(
        self,
        prompt: str,
        concept: Optional[OrthographicConceptPrompt] = None
    ) -> trimesh.Trimesh:
        archetype = self.detect_weapon_archetype(prompt)
        parts: List[trimesh.Trimesh] = []

        if archetype == WeaponArchetype.SWORD_BLADE:
            # Katana / Blade
            # Blade (curved box extending along +Y)
            blade = trimesh.creation.box(extents=[0.018, 0.82, 0.045])
            blade.apply_translation([0.0, 0.41, 0.0])
            parts.append(blade)

            # Tsuba (Oval guard disk)
            tsuba = trimesh.creation.cylinder(radius=0.055, height=0.012)
            rot_tsuba = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
            tsuba.apply_transform(rot_tsuba)
            tsuba.apply_translation([0.0, 0.0, 0.0])
            parts.append(tsuba)

            # Tsuka (Handle grip extending along -Y)
            handle = trimesh.creation.cylinder(radius=0.022, height=0.26)
            rot_handle = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
            handle.apply_transform(rot_handle)
            handle.apply_translation([0.0, -0.13, 0.0])
            parts.append(handle)

            # Pommel end cap
            pommel = trimesh.creation.icosphere(subdivisions=2, radius=0.026)
            pommel.apply_translation([0.0, -0.27, 0.0])
            parts.append(pommel)

        elif archetype == WeaponArchetype.HAMMER_BLUNT:
            # Shaft
            shaft = trimesh.creation.cylinder(radius=0.025, height=0.85)
            rot_shaft = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
            shaft.apply_transform(rot_shaft)
            shaft.apply_translation([0.0, 0.10, 0.0])
            parts.append(shaft)

            # Heavy Hammer Head
            head = trimesh.creation.box(extents=[0.14, 0.22, 0.16])
            head.apply_translation([0.0, 0.48, 0.0])
            parts.append(head)

        else:
            # Rifle / Gun / Blaster
            # Barrel
            barrel = trimesh.creation.cylinder(radius=0.018, height=0.62)
            rot_b = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
            barrel.apply_transform(rot_b)
            barrel.apply_translation([0.0, 0.42, 0.06])
            parts.append(barrel)

            # Upper & Lower Receiver
            receiver = trimesh.creation.box(extents=[0.065, 0.38, 0.10])
            receiver.apply_translation([0.0, 0.06, 0.06])
            parts.append(receiver)

            # Handguard / Shroud
            handguard = trimesh.creation.box(extents=[0.055, 0.32, 0.08])
            handguard.apply_translation([0.0, 0.30, 0.06])
            parts.append(handguard)

            # Pistol Grip (Angled back)
            grip = trimesh.creation.box(extents=[0.04, 0.06, 0.15])
            rot_g = trimesh.transformations.rotation_matrix(math.radians(18), [1, 0, 0])
            grip.apply_transform(rot_g)
            grip.apply_translation([0.0, -0.06, -0.06])
            parts.append(grip)

            # Magazine
            mag = trimesh.creation.box(extents=[0.035, 0.075, 0.16])
            mag.apply_translation([0.0, 0.08, -0.07])
            parts.append(mag)

            # Buttstock
            stock = trimesh.creation.box(extents=[0.048, 0.24, 0.12])
            stock.apply_translation([0.0, -0.24, 0.04])
            parts.append(stock)

            # Optic Scope
            scope = trimesh.creation.cylinder(radius=0.022, height=0.22)
            scope.apply_transform(rot_b)
            scope.apply_translation([0.0, 0.08, 0.14])
            parts.append(scope)

        combined = trimesh.util.concatenate(parts)
        combined.fix_normals()
        return combined
