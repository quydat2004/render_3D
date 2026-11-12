import pytest
import trimesh
import numpy as np
from services.inference.procedural_generator import (
    Procedural3DGenerator,
    CharacterArchetype,
    WeaponArchetype,
    _smin,
    _sdf_sphere,
    _sdf_capsule,
    _sdf_ellipsoid,
    _sdf_box
)
from services.orchestrator.gemini_orchestrator import OrthographicConceptPrompt

def test_sdf_primitives_and_smin():
    # 1. Sphere at (0, 0, 0) radius 1.0
    p = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]], dtype=np.float32)
    d_sphere = _sdf_sphere(p, np.array([0.0, 0.0, 0.0]), radius=1.0)
    assert np.isclose(d_sphere[0, 0], -1.0)  # center is -1.0
    assert np.isclose(d_sphere[0, 1], 1.0)   # distance 2.0 - 1.0 = 1.0

    # 2. Smooth minimum of two intersecting fields
    k = 0.1
    d1 = np.array([0.05], dtype=np.float32)
    d2 = np.array([0.05], dtype=np.float32)
    smooth_d = _smin(d1, d2, k=k)
    # smin(d, d, k) = d - k/4
    assert np.isclose(smooth_d[0], 0.05 - k / 4.0)

    # 3. Capsule along Z from 0 to 1 with radius 0.2
    p_cap = np.array([[[0.0, 0.0, 0.5]]], dtype=np.float32)
    d_cap = _sdf_capsule(p_cap, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), radius=0.2)
    assert np.isclose(d_cap[0, 0], -0.2)


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

def test_character_is_seamless_single_manifold():
    gen = Procedural3DGenerator()
    mesh = gen.generate_character_mesh("Cyberpunk samurai street warrior")
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 2000
    assert len(mesh.faces) > 4000
    # Seamless single manifold mesh: one unified connected component
    parts = mesh.split(only_watertight=False)
    assert len(parts) == 1, f"Expected 1 seamless component, got {len(parts)} separate disjoint pieces"

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
