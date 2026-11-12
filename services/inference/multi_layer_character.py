import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import trimesh
from skimage import measure

from services.inference.procedural_generator import (
    _smin,
    _smax,
    _sdf_sphere,
    _sdf_capsule,
    _sdf_ellipsoid,
    _sdf_box,
    CharacterArchetype
)

@dataclass
class MultiLayerCharacter:
    body: trimesh.Trimesh
    clothing: trimesh.Trimesh
    hair: trimesh.Trimesh

def _assign_cylindrical_uvs(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    v = mesh.vertices
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    u = np.zeros(len(v), dtype=np.float32)
    v_coord = np.zeros(len(v), dtype=np.float32)

    # Segment 1: Front face (Z >= 1.46, Y >= -0.02) gets dedicated 50% UV canvas area
    is_face = (z >= 1.46) & (y >= -0.02)
    u[is_face] = np.clip(0.5 + x[is_face] / 0.28, 0.05, 0.95)
    v_coord[is_face] = np.clip(0.5 + (z[is_face] - 1.61) / 0.28, 0.05, 0.95)

    # Segment 2: Rest of body
    is_body = ~is_face
    theta = np.arctan2(x[is_body], y[is_body])
    u[is_body] = np.clip((theta / (2.0 * np.pi)) + 0.5, 0.0, 1.0)
    z_min = float(np.min(z[is_body])) if np.any(is_body) else 0.0
    z_max = float(np.max(z[is_body])) if np.any(is_body) else 1.0
    height = max(z_max - z_min, 1e-5)
    v_coord[is_body] = np.clip((z[is_body] - z_min) / height, 0.0, 1.0)

    uv = np.stack([u, v_coord], axis=-1)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    return mesh

def make_ellipsoid(extents, subdivisions=2):
    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    s.vertices *= np.asarray(extents, dtype=np.float32) * 0.5
    return s

def create_detailed_head():
    parts = []
    # 1. Cranium (Braincase dome)
    cranium = trimesh.creation.icosphere(subdivisions=2, radius=0.115)
    cranium.vertices[:, 1] *= 1.05
    cranium.apply_translation([0.0, -0.015, 1.63])
    parts.append(cranium)

    # 2. Forehead & Supraorbital Brow Ridge
    brow = trimesh.creation.box(extents=[0.11, 0.035, 0.025])
    brow.apply_translation([0.0, 0.088, 1.645])
    parts.append(brow)

    # 3. Concave Orbital Sockets & Recessed Eyeballs (Left & Right)
    for sign in [-1, 1]:
        socket_rim = trimesh.creation.cylinder(radius=0.024, height=0.016, sections=14)
        rot_sock = trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0])
        socket_rim.apply_transform(rot_sock)
        socket_rim.apply_translation([sign * 0.042, 0.076, 1.622])
        parts.append(socket_rim)

        eyeball = trimesh.creation.icosphere(subdivisions=1, radius=0.016)
        eyeball.apply_translation([sign * 0.042, 0.082, 1.622])
        parts.append(eyeball)

        lid_u = trimesh.creation.cylinder(radius=0.017, height=0.010, sections=10)
        lid_u.apply_transform(rot_sock)
        lid_u.apply_translation([sign * 0.042, 0.085, 1.632])
        parts.append(lid_u)

        # Zygomatic Arch (Cheekbone structure)
        cheek = make_ellipsoid([0.036, 0.028, 0.022], subdivisions=1)
        cheek.apply_translation([sign * 0.065, 0.070, 1.585])
        parts.append(cheek)

    # 4. Sculpted 3D Nose (Nasal Bridge + Tip + Left/Right Nostril Alae)
    bridge = trimesh.creation.cone(radius=0.016, height=0.052, sections=12)
    rot_bridge = trimesh.transformations.rotation_matrix(math.radians(-75), [1, 0, 0])
    bridge.apply_transform(rot_bridge)
    bridge.apply_translation([0.0, 0.105, 1.61])
    parts.append(bridge)

    tip = trimesh.creation.icosphere(subdivisions=1, radius=0.014)
    tip.apply_translation([0.0, 0.128, 1.575])
    parts.append(tip)

    for sign in [-1, 1]:
        nostril = trimesh.creation.icosphere(subdivisions=1, radius=0.009)
        nostril.apply_translation([sign * 0.016, 0.118, 1.572])
        parts.append(nostril)

    # 5. Lips with Vermilion Border
    lip_u = make_ellipsoid([0.042, 0.018, 0.014], subdivisions=1)
    lip_u.apply_translation([0.0, 0.108, 1.542])
    parts.append(lip_u)

    lip_l = make_ellipsoid([0.038, 0.016, 0.015], subdivisions=1)
    lip_l.apply_translation([0.0, 0.102, 1.526])
    parts.append(lip_l)

    # 6. Chin (Mental Protuberance) & Mandible Jawline
    chin = make_ellipsoid([0.052, 0.040, 0.036], subdivisions=1)
    chin.apply_translation([0.0, 0.088, 1.488])
    parts.append(chin)

    # 7. Ears
    for sign in [-1, 1]:
        ear = make_ellipsoid([0.014, 0.032, 0.052], subdivisions=1)
        ear.apply_translation([sign * 0.112, -0.01, 1.61])
        parts.append(ear)

    head = trimesh.util.concatenate(parts)
    head.fix_normals()
    return head

def create_articulated_hand(is_left=True):
    parts = []
    sign = -1.0 if is_left else 1.0

    # Ergonomic palm with natural cupping (tapered ellipsoid palm)
    palm = make_ellipsoid([0.078, 0.088, 0.026], subdivisions=2)
    palm.apply_translation([0.0, 0.044, 0.0])
    parts.append(palm)

    # Thenar eminence (fleshy thumb base pad)
    thenar = make_ellipsoid([0.035, 0.048, 0.022], subdivisions=1)
    thenar.apply_translation([sign * 0.028, 0.028, 0.005])
    parts.append(thenar)

    # 1. Articulated Thumb (Metacarpal + Knuckle + Proximal + Joint + Distal)
    t_base = trimesh.creation.cylinder(radius=0.010, height=0.032, sections=12)
    t_base.apply_translation([0.0, 0.016, 0.0])
    t_mcp = trimesh.creation.icosphere(subdivisions=1, radius=0.011)
    t_mcp.apply_translation([0.0, 0.032, 0.0])
    t_prox = trimesh.creation.cylinder(radius=0.009, height=0.028, sections=12)
    t_prox.apply_translation([0.0, 0.046, 0.0])
    t_ip = trimesh.creation.icosphere(subdivisions=1, radius=0.0095)
    t_ip.apply_translation([0.0, 0.060, 0.0])
    t_dist = trimesh.creation.cylinder(radius=0.0075, height=0.024, sections=12)
    t_dist.apply_translation([0.0, 0.072, 0.0])
    t_tip = trimesh.creation.icosphere(subdivisions=1, radius=0.0075)
    t_tip.apply_translation([0.0, 0.084, 0.0])
    thumb_parts = trimesh.util.concatenate([t_base, t_mcp, t_prox, t_ip, t_dist, t_tip])

    rot_thumb = trimesh.transformations.rotation_matrix(math.radians(-sign * 42.0), [0, 0, 1])
    rot_thumb_fwd = trimesh.transformations.rotation_matrix(math.radians(22.0), [1, 0, 0])
    thumb_parts.apply_transform(rot_thumb_fwd)
    thumb_parts.apply_transform(rot_thumb)
    thumb_parts.apply_translation([sign * 0.040, 0.022, 0.006])
    parts.append(thumb_parts)

    # 2. Articulated 4 Fingers (Each with 3 Phalanx Segments + Knuckles)
    finger_specs = [
        ('index',  sign * 0.028, 0.072, 0.0085),
        ('middle', sign * 0.009, 0.080, 0.0090),
        ('ring',  -sign * 0.010, 0.074, 0.0085),
        ('pinky', -sign * 0.028, 0.060, 0.0075),
    ]

    for name, x_pos, length, rad in finger_specs:
        f_parts = []
        # MCP Knuckle Ball
        mcp = trimesh.creation.icosphere(subdivisions=1, radius=rad * 1.12)
        mcp.apply_translation([0.0, 0.0, 0.0])
        f_parts.append(mcp)

        # Phalanx 1 (Proximal, 42% of length)
        l1 = length * 0.42
        p1 = trimesh.creation.cylinder(radius=rad, height=l1, sections=10)
        p1.apply_translation([0.0, l1 * 0.5, 0.0])
        f_parts.append(p1)

        # PIP Knuckle Joint
        pip = trimesh.creation.icosphere(subdivisions=1, radius=rad * 0.98)
        pip.apply_translation([0.0, l1, 0.0])
        f_parts.append(pip)

        # Phalanx 2 (Intermediate, 33% of length)
        l2 = length * 0.33
        p2 = trimesh.creation.cylinder(radius=rad * 0.88, height=l2, sections=10)
        p2.apply_translation([0.0, l1 + l2 * 0.5, -0.0015])
        f_parts.append(p2)

        # DIP Knuckle Joint
        dip = trimesh.creation.icosphere(subdivisions=1, radius=rad * 0.82)
        dip.apply_translation([0.0, l1 + l2, -0.002])
        f_parts.append(dip)

        # Phalanx 3 (Distal, 25% of length) + Fingertip
        l3 = length * 0.25
        p3 = trimesh.creation.cylinder(radius=rad * 0.74, height=l3, sections=10)
        p3.apply_translation([0.0, l1 + l2 + l3 * 0.5, -0.0035])
        f_parts.append(p3)

        tip = trimesh.creation.icosphere(subdivisions=1, radius=rad * 0.74)
        tip.apply_translation([0.0, l1 + l2 + l3, -0.004])
        f_parts.append(tip)

        finger = trimesh.util.concatenate(f_parts)
        finger.apply_translation([x_pos, 0.085, 0.0])
        parts.append(finger)

    hand = trimesh.util.concatenate(parts)
    hand.fix_normals()
    return hand

def generate_hair_cards(
    head_center: np.ndarray = np.array([0.0, 0.0, 1.62], dtype=np.float32),
    num_cards: int = 44
) -> trimesh.Trimesh:
    cards_verts: List[np.ndarray] = []
    cards_faces: List[np.ndarray] = []
    cards_uvs: List[np.ndarray] = []

    vertex_offset = 0
    rng = np.random.RandomState(42)

    def _add_ribbon(points: List[np.ndarray], tangents: List[np.ndarray], widths: List[float]):
        nonlocal vertex_offset
        segs = len(points) - 1
        card_v = []
        card_uv = []
        for s in range(len(points)):
            t = s / float(segs)
            p = points[s]
            tx, ty = tangents[s]
            w = widths[s] * 0.5
            card_v.append([p[0] - tx * w, p[1] - ty * w, p[2]])
            card_v.append([p[0] + tx * w, p[1] + ty * w, p[2]])
            card_uv.append([0.0, t])
            card_uv.append([1.0, t])

        v_arr = np.array(card_v, dtype=np.float32)
        uv_arr = np.array(card_uv, dtype=np.float32)

        f_list = []
        for s in range(segs):
            i0 = vertex_offset + s * 2
            i1 = i0 + 1
            i2 = i0 + 2
            i3 = i0 + 3
            f_list.append([i0, i1, i2])
            f_list.append([i1, i3, i2])
            f_list.append([i0, i2, i1])
            f_list.append([i1, i2, i3])

        cards_verts.append(v_arr)
        cards_uvs.append(uv_arr)
        cards_faces.append(np.array(f_list, dtype=np.int32))
        vertex_offset += len(v_arr)

    # Cluster 1: Forehead Bangs / Fringe (8 ribbons covering forehead)
    num_bangs = 8
    for i in range(num_bangs):
        u = (i / float(num_bangs - 1)) - 0.5  # -0.5 to +0.5
        x_root = head_center[0] + u * 0.12
        y_root = head_center[1] + 0.088 - (u**2) * 0.03
        z_root = head_center[2] + 0.065
        bang_len = rng.uniform(0.08, 0.12)

        pts, tans, ws = [], [], []
        segs = 6
        for s in range(segs + 1):
            t = s / float(segs)
            z_c = z_root - t * bang_len
            # Gentle C-curve forward over forehead
            y_c = y_root + math.sin(t * math.pi * 0.7) * 0.025
            x_c = x_root + u * t * 0.02
            pts.append(np.array([x_c, y_c, z_c]))
            tans.append((1.0, 0.0))  # Horizontal width across brow
            ws.append(0.032 * (1.0 - t * 0.3))
        _add_ribbon(pts, tans, ws)

    # Cluster 2: Temples & Side Locks (14 ribbons flanking cheeks & jaw)
    num_sides = 14
    for sign in [-1, 1]:
        for i in range(num_sides // 2):
            t_side = i / float(num_sides // 2)
            angle = sign * (0.85 + t_side * 0.65)  # radians
            r_scalp = 0.128
            x_root = head_center[0] + r_scalp * math.sin(angle)
            y_root = head_center[1] + r_scalp * math.cos(angle) * 0.85
            z_root = head_center[2] + rng.uniform(0.02, 0.06)
            side_len = rng.uniform(0.40, 0.65)

            pts, tans, ws = [], [], []
            segs = 12
            for s in range(segs + 1):
                t = s / float(segs)
                z_c = z_root - t * side_len
                # S-curve along jawline and over shoulders
                sway = math.sin(t * math.pi * 1.5) * 0.025
                x_c = x_root + sign * (0.02 + t * 0.05) + sway
                y_c = y_root + (0.03 if y_root > 0 else -0.02) * math.sin(t * math.pi * 0.6)
                pts.append(np.array([x_c, y_c, z_c]))
                # Tangent along skull perimeter
                tx = -math.cos(angle)
                ty = math.sin(angle)
                tans.append((tx, ty))
                ws.append(0.045 * (1.0 - t * 0.25))
            _add_ribbon(pts, tans, ws)

    # Cluster 3: Back & Nape Volume (22 ribbons cascading down nape and upper back)
    num_back = 22
    for i in range(num_back):
        u_b = (i / float(num_back - 1)) - 0.5  # -0.5 to +0.5
        angle = math.pi + u_b * (math.pi * 0.75)  # Across the rear arc of skull
        r_scalp = 0.130 + rng.uniform(0.005, 0.020)
        x_root = head_center[0] + r_scalp * math.sin(angle)
        y_root = head_center[1] + r_scalp * math.cos(angle) * 0.90
        z_root = head_center[2] + rng.uniform(0.01, 0.08)
        back_len = rng.uniform(0.55, 0.85)

        pts, tans, ws = [], [], []
        segs = 12
        for s in range(segs + 1):
            t = s / float(segs)
            z_c = z_root - t * back_len
            # Outward drape away from spine
            sag_y = -math.sin(t * math.pi * 0.5) * 0.04
            flare_x = math.sin(t * math.pi * 0.5) * (u_b * 0.08)
            x_c = x_root + flare_x + rng.uniform(-0.01, 0.01) * math.sin(t * math.pi * 2.0)
            y_c = y_root + sag_y
            pts.append(np.array([x_c, y_c, z_c]))
            tx = -math.cos(angle)
            ty = math.sin(angle)
            tans.append((tx, ty))
            ws.append(0.052 * (1.0 - t * 0.30))
        _add_ribbon(pts, tans, ws)

    all_verts = np.vstack(cards_verts)
    all_uvs = np.vstack(cards_uvs)
    all_faces = np.vstack(cards_faces)

    hair_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
    hair_mesh.visual = trimesh.visual.TextureVisuals(uv=all_uvs)
    hair_mesh.fix_normals()
    return hair_mesh

def generate_plate_armor(archetype: str = "knight") -> trimesh.Trimesh:
    parts = []

    # 1. Sculpted Anatomical Cuirass (Convex Pectoral Armor + Sternum Ridge + Waist Taper)
    for sign in [-1, 1]:
        pec = make_ellipsoid([0.16, 0.14, 0.18], subdivisions=2)
        pec.vertices[pec.vertices[:, 1] < 0.0, 1] *= 0.3
        pec.apply_translation([sign * 0.082, 0.045, 1.28])
        parts.append(pec)

    # Central Sternum Ridge (Blade Deflector)
    sternum = trimesh.creation.box(extents=[0.022, 0.042, 0.22])
    sternum.apply_translation([0.0, 0.095, 1.27])
    parts.append(sternum)

    # Main Torso Backplate & Rib Encasing
    torso_core = trimesh.creation.cylinder(radius=0.20, height=0.34, sections=32)
    torso_core.vertices[:, 1] *= 0.72
    torso_core.apply_translation([0.0, -0.01, 1.22])
    parts.append(torso_core)

    # Gorget Flange (Neck Guard Rim)
    gorget = trimesh.creation.cylinder(radius=0.125, height=0.04, sections=24)
    gorget.apply_translation([0.0, 0.01, 1.39])
    parts.append(gorget)

    # 2. Articulated Pauldrons (Shoulder plate armor) & Vambraces
    arm_angle = math.radians(35.0)
    for sign in [-1, 1]:
        # Upper pauldron dome
        p1 = trimesh.creation.icosphere(subdivisions=2, radius=0.088)
        p1.vertices[:, 1] *= 0.85
        p1.apply_translation([sign * 0.19, 0.0, 1.36])
        parts.append(p1)

        # Lower pauldron rim plate
        p2 = trimesh.creation.cylinder(radius=0.080, height=0.08, sections=24)
        rot_p = trimesh.transformations.rotation_matrix(sign * arm_angle, [0, 1, 0])
        p2.apply_transform(rot_p)
        p2.apply_translation([sign * 0.23, 0.0, 1.28])
        parts.append(p2)

        # Vambraces (Forearm armor cuffs)
        elbow_x = sign * (0.175 + math.sin(arm_angle) * 0.27)
        elbow_z = 1.34 - math.cos(arm_angle) * 0.27
        vambrace = trimesh.creation.cylinder(radius=0.052, height=0.20, sections=20)
        vambrace.apply_transform(rot_p)
        vambrace.apply_translation([elbow_x + sign * math.sin(arm_angle) * 0.1, 0.0, elbow_z - math.cos(arm_angle) * 0.1])
        parts.append(vambrace)

    # 3. Faulds & Tassets (Hip plate armor skirt)
    faulds = trimesh.creation.cylinder(radius=0.215, height=0.18, sections=32)
    faulds.vertices[:, 1] *= 0.74
    scale_flare = 1.0 + (1.0 - (faulds.vertices[:, 2] / 0.18)) * 0.15
    faulds.vertices[:, :2] *= scale_flare[:, np.newaxis]
    faulds.apply_translation([0.0, 0.0, 0.98])
    parts.append(faulds)

    # 4. Greaves (Shin armor plates)
    leg_x = 0.105
    for sign in [-1, 1]:
        greave = trimesh.creation.cylinder(radius=0.070, height=0.34, sections=24)
        greave.vertices[:, 1] *= 0.82
        greave.apply_translation([sign * leg_x, 0.01, 0.26])
        parts.append(greave)

    armor_mesh = trimesh.util.concatenate(parts)
    return _assign_cylindrical_uvs(armor_mesh)

def generate_clothing_shell(
    archetype: str = "ghost_robed",
    num_rings: int = 32,
    radial_segments: int = 36,
    clothing_type: Optional[str] = None
) -> trimesh.Trimesh:
    c_type = clothing_type or ("dress" if "ghost" in str(archetype).lower() else "suit")
    is_robe_or_dress = c_type in ["dress", "robe", "kimono"]
    is_armor = c_type in ["armor", "knight", "mecha"]

    top_z = 1.38
    bottom_z = 0.08 if is_robe_or_dress else (0.75 if is_armor else 0.85)

    verts = []
    uvs = []
    faces = []

    for r in range(num_rings + 1):
        t_v = r / float(num_rings)  # 0 at neck/chest to 1 at bottom hem
        z_curr = top_z - t_v * (top_z - bottom_z)

        # Base radius along body height
        if is_robe_or_dress:
            # Flared bell-shaped flowing robe
            base_r = 0.22 + (t_v ** 1.8) * 0.32
        elif is_armor:
            # Flared hip faulds & chest cuirass
            base_r = 0.23 + (0.04 * math.sin(t_v * math.pi * 2.0))
        else:
            # Tactical coat / jacket shell
            base_r = 0.23 + (t_v * 0.06)

        for s in range(radial_segments):
            t_u = s / float(radial_segments)
            theta = t_u * math.pi * 2.0

            # Cloth wrinkles or armor facets
            if is_armor:
                folds = math.cos(theta * 8.0) * 0.015
            else:
                folds = math.sin(theta * 7.0) * (0.015 + t_v * 0.035)
                folds += math.cos(theta * 13.0) * (0.008 + t_v * 0.018)

            r_total = base_r + folds
            x = r_total * math.sin(theta)
            y = r_total * math.cos(theta) * 0.92  # Slight ellipse for human torso

            verts.append([x, y, z_curr])
            uvs.append([t_u, t_v])

    verts_arr = np.array(verts, dtype=np.float32)
    uvs_arr = np.array(uvs, dtype=np.float32)

    # Build faces
    for r in range(num_rings):
        for s in range(radial_segments):
            next_s = (s + 1) % radial_segments
            i0 = r * radial_segments + s
            i1 = r * radial_segments + next_s
            i2 = (r + 1) * radial_segments + s
            i3 = (r + 1) * radial_segments + next_s

            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
            # Backfaces for hollow cloth shell
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    faces_arr = np.array(faces, dtype=np.int32)
    cloth_mesh = trimesh.Trimesh(vertices=verts_arr, faces=faces_arr)
    cloth_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs_arr)
    cloth_mesh.fix_normals()
    return cloth_mesh

def generate_bony_base_body(archetype: str = "ghost_robed") -> trimesh.Trimesh:
    # High-detail anatomical body via SDF
    res_x, res_y, res_z = 84, 64, 130
    min_x, max_x = -0.55, 0.55
    min_y, max_y = -0.35, 0.35
    min_z, max_z = 0.0, 1.90

    vx = np.linspace(min_x, max_x, res_x, dtype=np.float32)
    vy = np.linspace(min_y, max_y, res_y, dtype=np.float32)
    vz = np.linspace(min_z, max_z, res_z, dtype=np.float32)
    X, Y, Z = np.meshgrid(vx, vy, vz, indexing='ij')
    P = np.stack([X, Y, Z], axis=-1)

    # 1. Head Cranium & Jaw (emaciated / bony)
    d_cranium = _sdf_ellipsoid(P, np.array([0.0, -0.01, 1.63], dtype=np.float32), np.array([0.115, 0.125, 0.13], dtype=np.float32))
    d_jaw = _sdf_ellipsoid(P, np.array([0.0, 0.04, 1.54], dtype=np.float32), np.array([0.075, 0.09, 0.08], dtype=np.float32))
    d_head = _smin(d_cranium, d_jaw, k=0.06)

    # Eye Sockets & Nose
    d_sock_l = _sdf_sphere(P, np.array([-0.045, 0.088, 1.625], dtype=np.float32), radius=0.028)
    d_sock_r = _sdf_sphere(P, np.array([0.045, 0.088, 1.625], dtype=np.float32), radius=0.028)
    d_head = _smax(d_head, -d_sock_l, k=0.02)
    d_head = _smax(d_head, -d_sock_r, k=0.02)

    d_nose_b = _sdf_capsule(P, np.array([0.0, 0.08, 1.64], dtype=np.float32), np.array([0.0, 0.125, 1.58], dtype=np.float32), radius=0.014)
    d_head = _smin(d_head, d_nose_b, k=0.02)

    # 2. Neck & Clavicle (Collarbone)
    d_neck = _sdf_capsule(P, np.array([0.0, 0.0, 1.40], dtype=np.float32), np.array([0.0, 0.0, 1.56], dtype=np.float32), radius=0.058)
    d_body = _smin(d_head, d_neck, k=0.05)

    d_clav_l = _sdf_capsule(P, np.array([0.0, 0.04, 1.41], dtype=np.float32), np.array([-0.16, 0.02, 1.41], dtype=np.float32), radius=0.016)
    d_clav_r = _sdf_capsule(P, np.array([0.0, 0.04, 1.41], dtype=np.float32), np.array([0.16, 0.02, 1.41], dtype=np.float32), radius=0.016)
    d_body = _smin(d_body, d_clav_l, k=0.025)
    d_body = _smin(d_body, d_clav_r, k=0.025)

    # 3. Torso + Bony Ribs
    d_chest = _sdf_ellipsoid(P, np.array([0.0, -0.01, 1.28], dtype=np.float32), np.array([0.18, 0.105, 0.16], dtype=np.float32))
    # Rib cage ridges
    for rib_z in [1.25, 1.20, 1.15]:
        d_rib_l = _sdf_ellipsoid(P, np.array([-0.10, 0.05, rib_z], dtype=np.float32), np.array([0.07, 0.03, 0.015], dtype=np.float32))
        d_rib_r = _sdf_ellipsoid(P, np.array([0.10, 0.05, rib_z], dtype=np.float32), np.array([0.07, 0.03, 0.015], dtype=np.float32))
        d_chest = _smin(d_chest, d_rib_l, k=0.02)
        d_chest = _smin(d_chest, d_rib_r, k=0.02)

    d_waist = _sdf_ellipsoid(P, np.array([0.0, 0.0, 1.05], dtype=np.float32), np.array([0.135, 0.095, 0.13], dtype=np.float32))
    d_hips = _sdf_ellipsoid(P, np.array([0.0, 0.0, 0.88], dtype=np.float32), np.array([0.155, 0.11, 0.13], dtype=np.float32))

    d_torso = _smin(_smin(d_chest, d_waist, k=0.07), d_hips, k=0.07)
    d_body = _smin(d_body, d_torso, k=0.07)

    # 4. Slender Arms in A-Pose & Bony Fingers
    arm_angle = math.radians(35.0)
    arm_len = 0.54
    sh_x = 0.175

    arm_l_start = np.array([-sh_x, 0.0, 1.34], dtype=np.float32)
    elbow_l = np.array([-sh_x - math.sin(arm_angle)*arm_len*0.5, 0.0, 1.34 - math.cos(arm_angle)*arm_len*0.5], dtype=np.float32)
    arm_l_end = np.array([-sh_x - math.sin(arm_angle)*arm_len, 0.0, 1.34 - math.cos(arm_angle)*arm_len], dtype=np.float32)

    d_arm_l = _smin(
        _sdf_capsule(P, arm_l_start, elbow_l, radius=0.046),
        _sdf_capsule(P, elbow_l, arm_l_end, radius=0.038),
        k=0.03
    )

    arm_r_start = np.array([sh_x, 0.0, 1.34], dtype=np.float32)
    elbow_r = np.array([sh_x + math.sin(arm_angle)*arm_len*0.5, 0.0, 1.34 - math.cos(arm_angle)*arm_len*0.5], dtype=np.float32)
    arm_r_end = np.array([sh_x + math.sin(arm_angle)*arm_len, 0.0, 1.34 - math.cos(arm_angle)*arm_len], dtype=np.float32)

    d_arm_r = _smin(
        _sdf_capsule(P, arm_r_start, elbow_r, radius=0.046),
        _sdf_capsule(P, elbow_r, arm_r_end, radius=0.038),
        k=0.03
    )

    d_body = _smin(d_body, d_arm_l, k=0.06)
    d_body = _smin(d_body, d_arm_r, k=0.06)

    # 5. Slender Legs
    leg_x = 0.105
    knee_z = 0.46
    knee_l = np.array([-leg_x, 0.0, knee_z], dtype=np.float32)
    knee_r = np.array([leg_x, 0.0, knee_z], dtype=np.float32)
    ankle_l = np.array([-leg_x, 0.0, 0.08], dtype=np.float32)
    ankle_r = np.array([leg_x, 0.0, 0.08], dtype=np.float32)

    d_leg_l = _smin(
        _sdf_capsule(P, np.array([-leg_x, 0.0, 0.86], dtype=np.float32), knee_l, radius=0.068),
        _sdf_capsule(P, knee_l, ankle_l, radius=0.056),
        k=0.035
    )
    d_leg_r = _smin(
        _sdf_capsule(P, np.array([leg_x, 0.0, 0.86], dtype=np.float32), knee_r, radius=0.068),
        _sdf_capsule(P, knee_r, ankle_r, radius=0.056),
        k=0.035
    )

    d_foot_l = _sdf_box(P, np.array([-leg_x, 0.04, 0.04], dtype=np.float32), np.array([0.08, 0.19, 0.07], dtype=np.float32))
    d_foot_r = _sdf_box(P, np.array([leg_x, 0.04, 0.04], dtype=np.float32), np.array([0.08, 0.19, 0.07], dtype=np.float32))

    d_body = _smin(d_body, _smin(d_leg_l, d_foot_l, k=0.03), k=0.06)
    d_body = _smin(d_body, _smin(d_leg_r, d_foot_r, k=0.03), k=0.06)

    # Marching cubes
    spacing = (vx[1] - vx[0], vy[1] - vy[0], vz[1] - vz[0])
    verts, faces, normals, _ = measure.marching_cubes(d_body, level=0.0, spacing=spacing)
    verts[:, 0] += vx[0]
    verts[:, 1] += vy[0]
    verts[:, 2] += vz[0]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.remove_unreferenced_vertices()

    split_parts = mesh.split(only_watertight=False)
    if len(split_parts) > 1:
        mesh = max(split_parts, key=lambda p: len(p.vertices))

    # Assemble Sculpted 3D Head and Articulated 5-Finger Hands
    head_mesh = create_detailed_head()

    hand_l = create_articulated_hand(is_left=True)
    rot_l = trimesh.transformations.rotation_matrix(arm_angle, [0, 1, 0])
    hand_l.apply_transform(rot_l)
    hand_l.apply_translation([
        -sh_x - math.sin(arm_angle) * (arm_len + 0.035),
        0.0,
        1.34 - math.cos(arm_angle) * (arm_len + 0.035)
    ])

    hand_r = create_articulated_hand(is_left=False)
    rot_r = trimesh.transformations.rotation_matrix(-arm_angle, [0, 1, 0])
    hand_r.apply_transform(rot_r)
    hand_r.apply_translation([
        sh_x + math.sin(arm_angle) * (arm_len + 0.035),
        0.0,
        1.34 - math.cos(arm_angle) * (arm_len + 0.035)
    ])

    full_body = trimesh.util.concatenate([mesh, head_mesh, hand_l, hand_r])
    full_body.remove_unreferenced_vertices()
    full_body.fix_normals()

    return _assign_cylindrical_uvs(full_body)
