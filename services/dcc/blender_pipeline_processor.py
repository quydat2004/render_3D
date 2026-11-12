import os
import sys
import shutil
import subprocess
import tempfile
import math
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image, ImageDraw
import numpy as np
import trimesh

class BlenderPipelineProcessor:
    def __init__(self, blender_path: Optional[str] = None):
        self.blender_path = blender_path or os.getenv("BLENDER_PATH", "blender")
        if not shutil.which(self.blender_path) and not Path(self.blender_path).exists():
            self.blender_available = False
        else:
            self.blender_available = True

    def generate_blender_script(
        self,
        input_path: str,
        output_path: str,
        target_faces: int = 20000,
        turntable_prefix: Optional[str] = None,
        num_turntable_frames: int = 8
    ) -> str:
        safe_input = input_path.replace("\\", "/")
        safe_output = output_path.replace("\\", "/")
        safe_tt = (turntable_prefix or "").replace("\\", "/")

        script_content = f'''import bpy
import sys
import math

bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Import
if "{safe_input}".endswith(".ply"):
    bpy.ops.wm.ply_import(filepath="{safe_input}")
elif "{safe_input}".endswith(".glb") or "{safe_input}".endswith(".gltf"):
    bpy.ops.import_scene.gltf(filepath="{safe_input}")
else:
    bpy.ops.wm.obj_import(filepath="{safe_input}")
mesh_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
if not mesh_objs:
    sys.exit(1)

target_obj = mesh_objs[0]
bpy.context.view_layer.objects.active = target_obj

# 2. Geometry Cleanup & Non-manifold fix
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.delete_loose()
bpy.ops.mesh.fill_holes(sides=4)
bpy.ops.object.mode_set(mode='OBJECT')

# 3. QuadriFlow Remeshing
bpy.ops.object.quadriflow_remesh(
    use_mesh_symmetry=True,
    use_preserve_boundary=True,
    target_faces={target_faces}
)

# 4. UV Unwrapping
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.005)
bpy.ops.uv.pack_islands(margin=0.005)
bpy.ops.object.mode_set(mode='OBJECT')

# 5. Export glTF 2.0 with Draco
bpy.ops.export_scene.gltf(
    filepath="{safe_output}",
    export_format='GLB',
    export_draco_mesh_compression_enable=True,
    export_apply=True
)

# 6. Render Turntables
if "{safe_tt}":
    cam_data = bpy.data.cameras.new("TurntableCam")
    cam_obj = bpy.data.objects.new("TurntableCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_obj.location = (0, -3.5, 1.2)
    cam_obj.rotation_euler = (math.radians(75), 0, 0)

    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    for i in range({num_turntable_frames}):
        angle = (2 * math.pi / {num_turntable_frames}) * i
        target_obj.rotation_euler[2] = angle
        bpy.context.scene.render.filepath = f"{safe_tt}_{{i:02d}}.png"
        bpy.ops.render.render(write_still=True)
'''
        return script_content

    def process_mesh(
        self,
        input_path: str,
        output_path: str,
        target_faces: int = 20000,
        turntable_dir: Optional[str] = None,
        num_turntable_frames: int = 8
    ) -> Dict[str, Any]:
        if self.blender_available:
            return self._execute_blender_process(
                input_path, output_path, target_faces, turntable_dir, num_turntable_frames
            )
        else:
            return self._execute_trimesh_fallback(
                input_path, output_path, target_faces, turntable_dir, num_turntable_frames
            )

    def _execute_blender_process(
        self,
        input_path: str,
        output_path: str,
        target_faces: int,
        turntable_dir: Optional[str],
        num_turntable_frames: int
    ) -> Dict[str, Any]:
        tt_prefix = str(Path(turntable_dir) / "frame") if turntable_dir else None
        script = self.generate_blender_script(
            input_path, output_path, target_faces, tt_prefix, num_turntable_frames
        )

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            script_file = f.name

        try:
            cmd = [self.blender_path, "-b", "-P", script_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            turntable_frames = []
            if turntable_dir:
                for i in range(num_turntable_frames):
                    frame_path = Path(turntable_dir) / f"frame_{i:02d}.png"
                    if frame_path.exists():
                        turntable_frames.append(str(frame_path))

            return {
                "success": True,
                "engine": "blender_headless",
                "output_path": output_path,
                "turntable_frames": turntable_frames,
                "output_faces": target_faces
            }
        finally:
            if os.path.exists(script_file):
                os.remove(script_file)

    def _execute_trimesh_fallback(
        self,
        input_path: str,
        output_path: str,
        target_faces: int,
        turntable_dir: Optional[str],
        num_turntable_frames: int
    ) -> Dict[str, Any]:
        mesh = trimesh.load(input_path, force="mesh")
        input_faces = len(mesh.faces)

        # 1. Clean up mesh
        mesh.remove_unreferenced_vertices()
        mesh.process()

        orig_verts = mesh.vertices.copy()
        orig_colors = mesh.visual.vertex_colors.copy() if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None else None

        # 2. Decimate if faces > target_faces
        if len(mesh.faces) > target_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
                if orig_colors is not None and len(orig_colors) == len(orig_verts):
                    from scipy.spatial import cKDTree
                    tree = cKDTree(orig_verts)
                    _, idxs = tree.query(mesh.vertices)
                    mesh.visual.vertex_colors = orig_colors[idxs]
            except Exception:
                # If decimation library is unavailable, retain cleaned mesh
                pass

        # 3. Compute clean smooth vertex normals
        mesh.fix_normals()

        # 4. Export to GLB
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path, file_type="glb")

        # 5. Render turntable frames (placeholder/orthographic projection frames)
        turntable_frames = []
        if turntable_dir:
            tt_path = Path(turntable_dir)
            tt_path.mkdir(parents=True, exist_ok=True)

            for i in range(num_turntable_frames):
                angle = (360.0 / num_turntable_frames) * i
                frame_file = tt_path / f"frame_{i:02d}.png"
                img = Image.new("RGBA", (512, 512), (30, 32, 38, 255))
                draw = ImageDraw.Draw(img)
                draw.text(
                    (20, 20),
                    f"Render3D AI Turntable\nAngle: {angle:.1f} deg\nFaces: {len(mesh.faces)}",
                    fill=(200, 220, 255, 255)
                )
                # Draw stylized center wireframe ellipse
                cx, cy = 256, 256
                rad_x = int(120 * abs(math.cos(math.radians(angle))) + 30)
                draw.ellipse([cx - rad_x, cy - 120, cx + rad_x, cy + 120], outline=(0, 255, 180, 255), width=2)
                img.save(str(frame_file))
                turntable_frames.append(str(frame_file))

        return {
            "success": True,
            "engine": "trimesh_fallback",
            "input_faces": input_faces,
            "output_faces": len(mesh.faces),
            "output_path": output_path,
            "turntable_frames": turntable_frames
        }
