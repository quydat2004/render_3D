import pytest
import numpy as np
import trimesh
from services.dcc.weapon_hard_surface import (
    WeaponHardSurfaceEngine,
    WeaponSocket,
    WeaponSocketType
)

def test_detect_weapon_sockets():
    engine = WeaponHardSurfaceEngine()

    # Create elongated rifle-like geometry
    # X: [-0.05, 0.05], Y (length): [-0.3, 0.7], Z (height): [-0.2, 0.15]
    gun_body = trimesh.creation.box(extents=[0.1, 1.0, 0.35])
    # Center is at [0, 0.2, 0] so Y goes from -0.3 to +0.7
    gun_body.apply_translation([0, 0.2, 0])

    sockets = engine.detect_sockets(gun_body)
    assert len(sockets) >= 3

    socket_dict = {s.name: s for s in sockets}
    assert "SOCKET_Muzzle" in socket_dict
    assert "SOCKET_Grip" in socket_dict
    assert "SOCKET_Optic" in socket_dict

    muzzle = socket_dict["SOCKET_Muzzle"]
    # Muzzle should be at the front (+Y direction, > 0.7)
    assert muzzle.position[1] >= 0.7

    optic = socket_dict["SOCKET_Optic"]
    # Optic should be at the top (+Z direction)
    assert optic.position[2] >= 0.15

def test_generate_collision_hull():
    engine = WeaponHardSurfaceEngine()
    mesh = trimesh.creation.cylinder(radius=0.1, height=1.0)

    ucx_mesh = engine.generate_collision_hull(mesh)
    assert isinstance(ucx_mesh, trimesh.Trimesh)
    assert ucx_mesh.is_convex is True
    assert len(ucx_mesh.faces) <= len(mesh.faces)

def test_generate_weapon_blender_script():
    engine = WeaponHardSurfaceEngine()
    script = engine.generate_weapon_blender_script(
        mesh_path="katana.obj",
        output_glb="katana_game_ready.glb"
    )
    assert "WEIGHTED_NORMAL" in script
    assert "keep_sharp" in script
    assert "SOCKET_Muzzle" in script
    assert "SOCKET_Grip" in script
    assert "UCX_" in script
