import pytest
import numpy as np
import trimesh
from services.inference.multi_layer_character import (
    generate_hair_cards,
    generate_clothing_shell,
    generate_plate_armor,
    generate_bony_base_body,
    MultiLayerCharacter
)

def test_generate_plate_armor():
    armor_mesh = generate_plate_armor(archetype="knight")
    assert isinstance(armor_mesh, trimesh.Trimesh)
    assert len(armor_mesh.vertices) > 200
    assert len(armor_mesh.faces) > 200
    bounds = armor_mesh.bounds
    width = bounds[1][0] - bounds[0][0]
    height = bounds[1][2] - bounds[0][2]
    assert width > 0.35
    assert height > 0.40

def test_generate_hair_cards():
    head_center = np.array([0.0, 0.0, 1.62], dtype=np.float32)
    hair_mesh = generate_hair_cards(head_center=head_center, num_cards=36)

    assert isinstance(hair_mesh, trimesh.Trimesh)
    assert len(hair_mesh.vertices) > 400
    assert len(hair_mesh.faces) > 200

    # Must have UVs defined for alpha hair texture
    assert hasattr(hair_mesh.visual, "uv")
    uvs = hair_mesh.visual.uv
    assert uvs is not None
    assert len(uvs) == len(hair_mesh.vertices)
    assert np.all(uvs >= -0.01) and np.all(uvs <= 1.01)

def test_generate_clothing_shell():
    cloth_mesh = generate_clothing_shell(archetype="ghost_robed")

    assert isinstance(cloth_mesh, trimesh.Trimesh)
    assert len(cloth_mesh.vertices) > 300
    assert len(cloth_mesh.faces) > 300

    # Must have UVs defined for cloth texture & alpha masking
    assert hasattr(cloth_mesh.visual, "uv")
    uvs = cloth_mesh.visual.uv
    assert uvs is not None
    assert len(uvs) == len(cloth_mesh.vertices)
    assert np.all(uvs >= -0.01) and np.all(uvs <= 1.01)

def test_generate_bony_base_body():
    body_mesh = generate_bony_base_body(archetype="ghost_robed")

    assert isinstance(body_mesh, trimesh.Trimesh)
    assert len(body_mesh.vertices) > 2000
    assert len(body_mesh.faces) > 4000
    assert hasattr(body_mesh.visual, "uv")
    uvs = body_mesh.visual.uv
    assert uvs is not None
    assert len(uvs) == len(body_mesh.vertices)
