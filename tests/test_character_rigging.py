import pytest
import numpy as np
import trimesh
from services.dcc.character_rigging_engine import (
    CharacterRiggingEngine,
    HumanoidLandmarks,
    STANDARD_HUMANOID_BONES
)

def test_standard_bones_list_has_ue5_hierarchy():
    assert "Pelvis" in STANDARD_HUMANOID_BONES
    assert "Head" in STANDARD_HUMANOID_BONES
    assert "Hand_L" in STANDARD_HUMANOID_BONES or "Wrist_L" in STANDARD_HUMANOID_BONES
    assert len(STANDARD_HUMANOID_BONES) >= 20

def test_estimate_humanoid_landmarks():
    engine = CharacterRiggingEngine()

    # Create sample humanoid-like mesh (cylinder representing body from z=0 to z=1.8m)
    humanoid_mesh = trimesh.creation.cylinder(radius=0.25, height=1.8)
    # Translate so feet are at z=0, head at z=1.8
    humanoid_mesh.apply_translation([0, 0, 0.9])

    landmarks = engine.estimate_landmarks(humanoid_mesh)
    assert isinstance(landmarks, HumanoidLandmarks)

    # Check vertical hierarchy
    assert landmarks.head[2] > landmarks.neck[2]
    assert landmarks.neck[2] > landmarks.spine[2]
    assert landmarks.spine[2] > landmarks.pelvis[2]
    assert landmarks.pelvis[2] > landmarks.knee_l[2]
    assert landmarks.knee_l[2] > landmarks.ankle_l[2]

    # Check lateral symmetry
    assert landmarks.shoulder_l[0] < 0  # left side
    assert landmarks.shoulder_r[0] > 0  # right side
    assert abs(abs(landmarks.shoulder_l[0]) - abs(landmarks.shoulder_r[0])) < 0.01

def test_clamp_skin_weights_max_4_influences():
    engine = CharacterRiggingEngine()

    # Raw unnormalized weights with 6 influences
    raw_weights = {
        "Pelvis": 0.40,
        "Spine": 0.30,
        "Thigh_L": 0.15,
        "Thigh_R": 0.10,
        "Calf_L": 0.05,
        "Foot_L": 0.02
    }

    clamped = engine.clamp_and_normalize_weights(raw_weights, max_influences=4)
    assert len(clamped) <= 4
    assert sum(clamped.values()) == pytest.approx(1.0, abs=1e-5)
    # The smallest 2 weights (Calf_L, Foot_L) should be dropped
    assert "Calf_L" not in clamped
    assert "Foot_L" not in clamped
    assert "Pelvis" in clamped

def test_generate_rigging_script():
    engine = CharacterRiggingEngine()
    script = engine.generate_rigging_script(
        mesh_path="char.obj",
        output_glb="rigged_char.glb"
    )
    assert "bpy.data.armatures.new" in script
    assert "ARMATURE_AUTO" in script
    assert "Pelvis" in script
    assert "clamp_vertex_weights" in script
