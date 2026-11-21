import pytest
from pathlib import Path
import trimesh
from services.dcc.character_rigging_engine import CharacterRiggingEngine

def test_rig_and_animate_character_creates_glb(tmp_path):
    engine = CharacterRiggingEngine()

    # Head sits above a wide torso, so the nearest bones to a head vertex include the shoulders.
    # Distance weighting alone binds the face to the arm bones.
    import numpy as np
    head = trimesh.creation.icosphere(subdivisions=2, radius=0.16)
    head.apply_translation([0.0, 1.55, 0.0])
    torso = trimesh.creation.box(extents=(0.9, 0.7, 0.3))
    torso.apply_translation([0.0, 1.0, 0.0])
    mesh = trimesh.util.concatenate([head, torso])
    input_glb = str(tmp_path / "test_char.glb")
    mesh.export(input_glb)

    output_glb = str(tmp_path / "test_char_animated.glb")
    res_path = engine.rig_and_animate_character(
        input_glb=input_glb,
        output_glb=output_glb,
        animations=["walk", "idle", "run", "attack"]
    )

    assert Path(res_path).exists()
    assert Path(res_path).stat().st_size > 1000

    # Read glb structure
    import json
    import struct
    data = Path(res_path).read_bytes()
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67  # 'glTF'
    assert version == 2

    chunk_len, chunk_type = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20+chunk_len].decode("utf-8"))

    # Verify skin and animations exist
    assert "skins" in gltf
    assert len(gltf["skins"]) > 0
    assert "animations" in gltf
    assert len(gltf["animations"]) >= 4

    anim_names = [a.get("name", "").lower() for a in gltf["animations"]]
    assert "walk" in anim_names
    assert "idle" in anim_names
    assert "run" in anim_names
    assert "attack" in anim_names

    # Head vertices must be skinned only to the Neck and Head bones. Any weight on a limb bone makes
    # the face stretch when the arms or legs swing.
    import numpy as np
    HEAD, NECK = 4, 3
    limb_bones = {5, 6, 7, 8, 9, 10, 11, 12}
    skin = gltf["skins"][0]
    prim = gltf["meshes"][0]["primitives"][0]
    bin_start = 20 + chunk_len + 8

    def _acc(idx):
        acc = gltf["accessors"][idx]
        bv = gltf["bufferViews"][acc["bufferView"]]
        off = bin_start + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return acc, off

    pos_acc, pos_off = _acc(prim["attributes"]["POSITION"])
    pos = np.frombuffer(data[pos_off:pos_off + pos_acc["count"] * 12], dtype=np.float32).reshape(-1, 3)
    j_acc, j_off = _acc(prim["attributes"]["JOINTS_0"])
    joints = np.frombuffer(data[j_off:j_off + j_acc["count"] * 8], dtype=np.uint16).reshape(-1, 4)
    w_acc, w_off = _acc(prim["attributes"]["WEIGHTS_0"])
    weights = np.frombuffer(data[w_off:w_off + w_acc["count"] * 16], dtype=np.float32).reshape(-1, 4)

    y = pos[:, 1]
    head_band = y > (y.min() + (y.max() - y.min()) * 0.85)
    assert head_band.any()
    for vi in np.where(head_band)[0]:
        for slot in range(4):
            if weights[vi, slot] > 0.01:
                assert int(joints[vi, slot]) in (NECK, HEAD), \
                    f"head vertex {vi} weighted {weights[vi, slot]:.2f} to limb bone {int(joints[vi, slot])}"

    # The Hips node parents the whole spine, neck and head. A translation track on it lifts and
    # drops the entire upper body, which reads as the face bobbing. Limbs rotate; the torso stays put.
    node_names = [n.get("name") for n in gltf["nodes"]]
    hips_node = node_names.index("Hips")
    for clip in gltf["animations"]:
        for ch in clip["channels"]:
            tgt = ch["target"]
            assert not (tgt["node"] == hips_node and tgt["path"] == "translation"), \
                f"{clip['name']} translates the Hips node, which drags the head up and down"

def _raw_gltf_uv(glb_path):
    """Reads TEXCOORD_0 exactly as stored in the GLB file (no trimesh V-flip)."""
    import json
    import struct
    import numpy as np
    data = Path(glb_path).read_bytes()
    chunk_len, _ = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20 + chunk_len].decode("utf-8"))
    prim = gltf["meshes"][0]["primitives"][0]
    acc = gltf["accessors"][prim["attributes"]["TEXCOORD_0"]]
    bv = gltf["bufferViews"][acc["bufferView"]]
    bin_start = 20 + chunk_len + 8
    off = bin_start + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.frombuffer(data[off:off + acc["count"] * 8], dtype=np.float32).reshape(-1, 2)

def test_rig_and_animate_preserves_uv_orientation_of_input_glb(tmp_path):
    """
    Hunyuan3D-Paint UVs are authored for the file's own convention. The rigged GLB must keep the
    exact same per-vertex UVs as the input FILE (trimesh flips V on load, so the writer must flip back).
    """
    from PIL import Image
    import numpy as np

    engine = CharacterRiggingEngine()
    box = trimesh.creation.box(extents=(0.6, 0.3, 1.75))
    # trimesh/OpenGL convention in memory: V=1 is the top of the texture. Head vertices get V=1.
    in_memory_uv = np.zeros((len(box.vertices), 2), dtype=np.float32)
    in_memory_uv[:, 1] = np.where(box.vertices[:, 1] > 0, 0.9, 0.1)
    box.visual = trimesh.visual.TextureVisuals(uv=in_memory_uv, image=Image.new("RGB", (32, 32), (200, 30, 30)))
    input_glb = str(tmp_path / "painted_in.glb")
    box.export(input_glb)

    # Vertex order can change on load; compare per-vertex via position keys.
    in_loaded = list(trimesh.load(input_glb).geometry.values())[0]
    raw_in = _raw_gltf_uv(input_glb)
    assert len(raw_in) == len(in_loaded.vertices)

    output_glb = str(tmp_path / "painted_out.glb")
    engine.rig_and_animate_character(input_glb, output_glb)
    raw_out = _raw_gltf_uv(output_glb)

    # Head = top half of the mesh. Raw-file V at the head must match the input file's raw V at the head.
    head_in = raw_in[in_loaded.vertices[:, 1] > in_loaded.bounds.mean(axis=0)[1], 1].mean()
    out_mesh = list(trimesh.load(output_glb).geometry.values())[0]
    head_out = raw_out[out_mesh.vertices[:, 1] > out_mesh.bounds.mean(axis=0)[1], 1].mean()
    assert abs(head_in - head_out) < 0.05, f"UV V inverted by rigging: input head V={head_in:.2f}, output head V={head_out:.2f}"


def test_rig_and_animate_embeds_materials_and_textures(tmp_path):
    from PIL import Image
    import numpy as np
    import json
    import struct

    engine = CharacterRiggingEngine()
    box = trimesh.creation.box(extents=(0.6, 0.3, 1.75))
    tex_img = Image.new("RGB", (64, 64), (220, 30, 30))
    uvs = np.zeros((len(box.vertices), 2), dtype=np.float32)
    box.visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_img)
    input_glb = str(tmp_path / "textured_in.glb")
    box.export(input_glb)

    output_glb = str(tmp_path / "textured_out.glb")
    engine.rig_and_animate_character(input_glb, output_glb)

    data = Path(output_glb).read_bytes()
    chunk_len, _ = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20+chunk_len].decode("utf-8"))

    assert "materials" in gltf, "glTF must preserve materials"
    assert len(gltf["materials"]) > 0
    assert "textures" in gltf, "glTF must preserve textures"
    assert "images" in gltf, "glTF must embed texture images"
    assert gltf["meshes"][0]["primitives"][0].get("material") is not None
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 0.0, "Character skin/cloth must have 0.0 metalness to prevent plastic/chrome shine"
    assert 0.6 <= pbr["roughnessFactor"] <= 0.7, "Character skin/cloth roughness must be in 0.6..0.7 matte range"
    assert "metallicRoughnessTexture" in pbr, "Character must embed zone-separated metallicRoughnessTexture"
    assert "occlusionTexture" in gltf["materials"][0], "glTF must embed ambient occlusion map"


def test_rig_and_animate_preserves_multi_layer_primitives(tmp_path):
    from PIL import Image
    import numpy as np
    import json
    import struct

    engine = CharacterRiggingEngine()
    scene = trimesh.Scene()

    body = trimesh.creation.box(extents=(0.5, 1.75, 0.3))
    body.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(body.vertices), 2), dtype=np.float32),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (32, 32), (230, 190, 165)),
            metallicFactor=0.0,
            roughnessFactor=0.55
        )
    )
    armor = trimesh.creation.box(extents=(0.55, 0.6, 0.35))
    armor.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(armor.vertices), 2), dtype=np.float32),
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.83, 0.68, 0.21, 1.0],
            metallicFactor=0.95,
            roughnessFactor=0.18
        )
    )
    hair = trimesh.creation.box(extents=(0.25, 0.3, 0.25))
    hair.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(hair.vertices), 2), dtype=np.float32),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGBA", (32, 32), (20, 20, 25, 200)),
            alphaMode="MASK",
            alphaCutoff=0.5
        )
    )

    scene.add_geometry(body, node_name="Mesh_Body", geom_name="Mesh_Body")
    scene.add_geometry(armor, node_name="Mesh_Armor", geom_name="Mesh_Armor")
    scene.add_geometry(hair, node_name="Mesh_HairCards", geom_name="Mesh_HairCards")

    in_glb = str(tmp_path / "multi_in.glb")
    scene.export(in_glb)

    out_glb = str(tmp_path / "multi_out.glb")
    engine.rig_and_animate_character(in_glb, out_glb)

    data = Path(out_glb).read_bytes()
    chunk_len, _ = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20+chunk_len].decode("utf-8"))

    prims = gltf["meshes"][0]["primitives"]
    assert len(prims) == 3, "All 3 character layers (Body, Armor, Hair) must be preserved as distinct primitives"
    assert len(gltf["materials"]) == 3
    assert any(m.get("alphaMode") == "MASK" for m in gltf["materials"])
    assert any(m["pbrMetallicRoughness"].get("metallicFactor", 0) > 0.8 for m in gltf["materials"])

def test_rig_and_animate_chibi_morphology_and_skin_isolation(tmp_path):
    import json
    import struct
    import numpy as np

    engine = CharacterRiggingEngine()

    # Create a Chibi character: large head (radius 0.30 at y=1.2), short body (y=0.4..0.8), pre-bent knees at z=0.08
    head = trimesh.creation.icosphere(subdivisions=2, radius=0.30)
    head.apply_translation([0.0, 1.25, 0.0])

    torso = trimesh.creation.cylinder(radius=0.18, height=0.40, sections=24)
    torso.apply_translation([0.0, 0.70, 0.0])

    # Left and right legs with pre-bent knees protruding in +Z
    leg_l = trimesh.creation.box(extents=(0.10, 0.45, 0.14))
    leg_l.apply_translation([-0.10, 0.25, 0.06]) # forward Z bend

    leg_r = trimesh.creation.box(extents=(0.10, 0.45, 0.14))
    leg_r.apply_translation([0.10, 0.25, 0.06])

    # Arms close to the body
    arm_l = trimesh.creation.box(extents=(0.06, 0.30, 0.06))
    arm_l.apply_translation([-0.25, 0.65, 0.0])

    arm_r = trimesh.creation.box(extents=(0.06, 0.30, 0.06))
    arm_r.apply_translation([0.25, 0.65, 0.0])

    chibi_mesh = trimesh.util.concatenate([head, torso, leg_l, leg_r, arm_l, arm_r])
    input_glb = str(tmp_path / "chibi_in.glb")
    chibi_mesh.export(input_glb)

    output_glb = str(tmp_path / "chibi_out.glb")
    engine.rig_and_animate_character(input_glb, output_glb)

    data = Path(output_glb).read_bytes()
    chunk_len, _ = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20+chunk_len].decode("utf-8"))

    # 1. Bone placement: LowerLeg bones (bones 9 and 11, index 10 and 12)
    # The knee position should reflect pre-bent knee Z offset > 0
    node_names = [n["name"] for n in gltf["nodes"]]
    lower_l_node = gltf["nodes"][node_names.index("LowerLeg.L")]
    assert "translation" in lower_l_node

    # 2. Verify Torso vertices are strictly isolated from arm bones (bones 5..8)
    prim = gltf["meshes"][0]["primitives"][0]
    bin_start = 20 + chunk_len + 8
    def _acc(idx):
        acc = gltf["accessors"][idx]
        bv = gltf["bufferViews"][acc["bufferView"]]
        off = bin_start + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return acc, off

    pos_acc, pos_off = _acc(prim["attributes"]["POSITION"])
    pos = np.frombuffer(data[pos_off:pos_off + pos_acc["count"] * 12], dtype=np.float32).reshape(-1, 3)
    j_acc, j_off = _acc(prim["attributes"]["JOINTS_0"])
    joints = np.frombuffer(data[j_off:j_off + j_acc["count"] * 8], dtype=np.uint16).reshape(-1, 4)
    w_acc, w_off = _acc(prim["attributes"]["WEIGHTS_0"])
    weights = np.frombuffer(data[w_off:w_off + w_acc["count"] * 16], dtype=np.float32).reshape(-1, 4)

    # Core torso vertices: central x (abs(x) < 0.10) between hip and chest (y in [0.55, 0.85])
    torso_mask = (np.abs(pos[:, 0]) < 0.10) & (pos[:, 1] >= 0.55) & (pos[:, 1] <= 0.85)
    assert np.any(torso_mask)
    arm_bone_indices = {5, 6, 7, 8} # UpperArm.L, Forearm.L, UpperArm.R, Forearm.R
    for vi in np.where(torso_mask)[0]:
        for slot in range(4):
            if weights[vi, slot] > 0.05:
                assert int(joints[vi, slot]) not in arm_bone_indices, \
                    f"Torso vertex {vi} has {weights[vi, slot]:.2f} weight on arm bone {int(joints[vi, slot])}"

    # 3. Verify Left vs Right lateral isolation: left limb vertices shouldn't be weighted to right bones
    right_bone_indices = {7, 8, 11, 12}
    left_limb_mask = (pos[:, 0] < -0.15) & (pos[:, 1] < 1.0)
    assert np.any(left_limb_mask)
    for vi in np.where(left_limb_mask)[0]:
        for slot in range(4):
            if weights[vi, slot] > 0.05:
                assert int(joints[vi, slot]) not in right_bone_indices, \
                    f"Left limb vertex {vi} weighted to right bone {int(joints[vi, slot])}"

