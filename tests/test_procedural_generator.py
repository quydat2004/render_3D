import pytest
import trimesh
import numpy as np
from services.inference.procedural_generator import (
    Procedural3DGenerator,
    CharacterArchetype,
    WeaponArchetype
)
from services.orchestrator.gemini_orchestrator import OrthographicConceptPrompt

def test_detect_character_archetype():
    gen = Procedural3DGenerator()
    assert gen.detect_character_archetype("Ma nữ kinh dị châu Á áo dài trắng") == CharacterArchetype.GHOST_ROBED
    assert gen.detect_character_archetype("Cyberpunk street samurai warrior") == CharacterArchetype.WARRIOR_HUMANOID
    assert gen.detect_character_archetype("Heavy mecha robot titan") == CharacterArchetype.MECHA_HEAVY

def test_detect_weapon_archetype():
    gen = Procedural3DGenerator()
    assert gen.detect_weapon_archetype("Cyberpunk katana neon sword") == WeaponArchetype.SWORD_BLADE
    assert gen.detect_weapon_archetype("Laser sniper rifle with scope") == WeaponArchetype.RIFLE_GUN
    assert gen.detect_weapon_archetype("Battle war hammer") == WeaponArchetype.HAMMER_BLUNT

def test_character_has_face_features_and_vertex_colors():
    gen = Procedural3DGenerator()
    concept = OrthographicConceptPrompt(
        front_prompt="Ma nữ kinh dị tóc dài áo trắng",
        side_prompt="Side view",
        back_prompt="Back view",
        style_notes="Ghostly pale skin, black hair, red eyes",
        pose="A-pose",
        poly_budget=20000
    )

    mesh = gen.generate_character_mesh("Ma nữ kinh dị châu Á", concept)
    assert isinstance(mesh, trimesh.Trimesh)

    # 1. Height along Y-up should be realistic
    bounds = mesh.bounds
    height = bounds[1][1] - bounds[0][1]
    assert 1.4 <= height <= 2.1

    # 2. Check vertex colors exist
    assert hasattr(mesh.visual, "vertex_colors")
    colors = mesh.visual.vertex_colors
    assert colors is not None
    assert len(colors) == len(mesh.vertices)

    # 3. Check for multi-color variety (skin, black hair, eyes, dress)
    unique_colors = np.unique(colors[:, :3], axis=0)
    assert len(unique_colors) >= 3  # Must have at least 3 distinct color regions

def test_weapon_has_distinct_material_vertex_colors():
    gen = Procedural3DGenerator()
    concept = OrthographicConceptPrompt(
        front_prompt="Cyberpunk Katana",
        side_prompt="Side profile",
        back_prompt="Rear view",
        style_notes="Metallic blade with gold guard and dark grip",
        pose="Neutral",
        poly_budget=8000
    )

    mesh = gen.generate_weapon_mesh("Cyberpunk Katana neon sword", concept)
    assert isinstance(mesh, trimesh.Trimesh)

    # Check vertex colors for weapon (blade != guard != handle)
    assert hasattr(mesh.visual, "vertex_colors")
    colors = mesh.visual.vertex_colors
    unique_colors = np.unique(colors[:, :3], axis=0)
    assert len(unique_colors) >= 3
