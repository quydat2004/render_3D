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

def test_generate_character_humanoid_mesh():
    gen = Procedural3DGenerator()
    concept = OrthographicConceptPrompt(
        front_prompt="Ma nữ kinh dị tóc dài áo trắng",
        side_prompt="Side view",
        back_prompt="Back view",
        style_notes="Ghostly pale",
        pose="A-pose",
        poly_budget=20000
    )

    mesh = gen.generate_character_mesh("Ma nữ kinh dị châu Á", concept)
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 200
    assert len(mesh.faces) > 200

    # Check vertical height is realistic human scale (1.5m to 2.0m along Y-up)
    bounds = mesh.bounds
    height = bounds[1][1] - bounds[0][1]
    assert 1.4 <= height <= 2.1

    # Check A-pose lateral extension (arms should extend sideways |x| >= 0.3m)
    span_x = bounds[1][0] - bounds[0][0]
    assert span_x >= 0.6  # Width should be at least 0.6m across arms

def test_generate_weapon_katana_mesh():
    gen = Procedural3DGenerator()
    concept = OrthographicConceptPrompt(
        front_prompt="Cyberpunk Katana",
        side_prompt="Side profile",
        back_prompt="Rear view",
        style_notes="Metallic blade with cyan edge",
        pose="Neutral",
        poly_budget=8000
    )

    mesh = gen.generate_weapon_mesh("Cyberpunk Katana neon sword", concept)
    assert isinstance(mesh, trimesh.Trimesh)
    bounds = mesh.bounds
    length = max(bounds[1] - bounds[0])
    # Swords should be 0.8m to 1.3m long
    assert 0.7 <= length <= 1.4

def test_generate_weapon_rifle_mesh():
    gen = Procedural3DGenerator()
    concept = OrthographicConceptPrompt(
        front_prompt="Sniper rifle",
        side_prompt="Profile",
        back_prompt="Stock",
        style_notes="Gunmetal",
        pose="Neutral",
        poly_budget=8000
    )

    mesh = gen.generate_weapon_mesh("Futuristic sniper rifle with scope", concept)
    assert isinstance(mesh, trimesh.Trimesh)
    bounds = mesh.bounds
    # Rifle should be elongated and have vertical height from grip/scope
    assert bounds[1][1] - bounds[0][1] >= 0.6  # length >= 0.6m
    assert bounds[1][2] - bounds[0][2] >= 0.15 # height >= 0.15m
