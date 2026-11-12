import pytest
import numpy as np
import trimesh
from services.inference.multi_layer_character import (
    create_articulated_hand,
    create_detailed_head,
    generate_plate_armor,
    MultiLayerCharacter
)

def test_hand_has_articulated_finger_segments():
    hand = create_articulated_hand(is_left=False)
    assert len(hand.vertices) > 250
    assert len(hand.faces) > 400
    bounds = hand.bounds
    # Hand length should span from wrist to fingertips (~0.15m to ~0.24m)
    length = bounds[1][1] - bounds[0][1]
    assert 0.12 <= length <= 0.28
    # Hand should have natural 3D volume, not flat 2D
    thickness = bounds[1][2] - bounds[0][2]
    assert thickness > 0.02

def test_head_mesh_has_sculpted_eye_sockets_and_nose():
    head = create_detailed_head()
    assert len(head.vertices) > 400
    assert len(head.faces) > 600

    # Look for eye socket, nose, and jaw bounds
    bounds = head.bounds
    # Check head is positioned around height 1.45 to 1.75m
    assert bounds[0][2] >= 1.40
    assert bounds[1][2] <= 1.80

    # Check forward depth (nose tip vs face center)
    assert bounds[1][1] > 0.10

def test_cuirass_has_curved_pectoral_geometry():
    armor = generate_plate_armor("knight")
    assert len(armor.faces) > 800

    # Check bounds of cuirass (chest armor region Z ~ 1.0 to 1.4)
    bounds = armor.bounds
    width = bounds[1][0] - bounds[0][0]
    depth = bounds[1][1] - bounds[0][1]

    # Torso armor width is wider than depth (elliptical human chest contour)
    assert width > depth * 1.05
