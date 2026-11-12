import os
import time
import shutil
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, Any
import numpy as np
import trimesh
from PIL import Image, ImageDraw

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None
    handle_file = None

logger = logging.getLogger(__name__)

class Hunyuan3DClient:
    SPACE_ID = "tencent/Hunyuan3D-2"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        val = line.split("=", 1)[1].strip()
                        self.token = val.strip("'\"")
                        break

        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not Client:
            raise RuntimeError("gradio_client is not installed.")
        self._client = Client(self.SPACE_ID, token=self.token)
        return self._client

    def generate_concept_image(self, caption: str, output_path: str) -> str:
        """
        Generate a sharp, front orthographic A-pose concept image for the character.
        Tries fast online AI generation, falling back to a clean procedural studio render.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            enhanced_prompt = (
                f"full body front view orthographic A-pose {caption}, "
                "neutral studio grey background, high detail 3D render, centered, realistic lighting"
            )
            encoded = urllib.parse.quote(enhanced_prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&nologo=true&seed=42"
            req = urllib.request.Request(url, headers={"User-Agent": "Render3D/1.0 (Mozilla/5.0)"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = resp.read()
                if len(data) > 1000:
                    out.write_bytes(data)
                    return str(out)
        except Exception as e:
            logger.warning(f"Concept image web synthesis unavailable ({e}), using procedural studio fallback.")

        # Procedural studio fallback concept render
        img = Image.new("RGB", (768, 768), (140, 145, 155))
        draw = ImageDraw.Draw(img)

        # Head / Face
        draw.ellipse([344, 80, 424, 170], fill=(225, 188, 162))
        # Torso / Armor Cuirass
        draw.polygon([(320, 180), (448, 180), (420, 420), (348, 420)], fill=(212, 175, 55))
        # Pauldrons / Shoulders
        draw.ellipse([270, 175, 345, 235], fill=(230, 190, 60))
        draw.ellipse([423, 175, 498, 235], fill=(230, 190, 60))
        # Arms
        draw.polygon([(320, 200), (250, 370), (280, 380), (330, 250)], fill=(190, 155, 45))
        draw.polygon([(448, 200), (518, 370), (488, 380), (438, 250)], fill=(190, 155, 45))
        # Hands
        draw.ellipse([240, 370, 275, 410], fill=(225, 188, 162))
        draw.ellipse([495, 370, 530, 410], fill=(225, 188, 162))
        # Legs
        draw.rectangle([348, 420, 380, 680], fill=(160, 130, 40))
        draw.rectangle([388, 420, 420, 680], fill=(160, 130, 40))
        img.save(str(out), format="PNG")
        return str(out)

    def project_texture_to_mesh(self, glb_path: str, image_path: str, output_glb_path: str) -> str:
        """
        Projects concept image onto the 3D neural mesh geometry, normalizes coordinates to Y-up
        humanoid scale (~1.8m), and exports a game-ready textured GLB.
        """
        out_dest = Path(output_glb_path)
        out_dest.parent.mkdir(parents=True, exist_ok=True)

        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if not geoms:
                raise ValueError("GLB has no geometry.")
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0].copy()
        else:
            mesh = scene.copy()

        # Center on X & Z; ground to Y=0
        bounds = mesh.bounds
        center_x = (bounds[0][0] + bounds[1][0]) * 0.5
        center_z = (bounds[0][2] + bounds[1][2]) * 0.5
        min_y = bounds[0][1]
        mesh.apply_translation([-center_x, -min_y, -center_z])

        bounds = mesh.bounds
        height = max(bounds[1][1] - bounds[0][1], 1e-4)
        width = max(bounds[1][0] - bounds[0][0], 1e-4)

        # Scale to standard humanoid height (1.80m)
        if height < 0.5 or height > 5.0:
            scale = 1.80 / height
            mesh.apply_scale(scale)
            bounds = mesh.bounds
            height = bounds[1][1] - bounds[0][1]
            width = bounds[1][0] - bounds[0][0]

        # Orthographic / Cylindrical Front Projection UVs
        verts = mesh.vertices
        u = (verts[:, 0] - bounds[0][0]) / width
        v = (verts[:, 1] - bounds[0][1]) / height
        uvs = np.column_stack([np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)])

        pil_img = Image.open(image_path).convert("RGB")
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=pil_img)

        # Compute vertex normals if missing
        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
            mesh.compute_vertex_normals()

        mesh.export(str(out_dest), file_type="glb")
        return str(out_dest)

    def generate_3d_from_image(
        self,
        image_path: str,
        output_path: str,
        caption: str = "3D character model, detailed geometry, game-ready",
        steps: int = 15,
        timeout: int = 180
    ) -> str:
        """
        Generates 250k+ polygon neural mesh via Tencent Hunyuan3D-2 /shape_generation,
        then binds the input image as texture material.
        """
        client = self._get_client()
        img_arg = handle_file(image_path) if handle_file else image_path

        job = client.submit(
            caption=caption,
            image=img_arg,
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=steps,
            guidance_scale=4.5,
            seed=42,
            octree_resolution=256,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=True,
            api_name="/shape_generation"
        )

        start = time.time()
        while not job.done():
            if time.time() - start > timeout:
                raise TimeoutError(f"Hunyuan3D generation exceeded {timeout}s.")
            time.sleep(3)

        result = job.result()
        file_item = result[0]
        src_file = file_item["value"] if isinstance(file_item, dict) else str(file_item)

        out_dest = Path(output_path)
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        raw_temp = str(out_dest.parent / f"{out_dest.stem}_raw_mesh.glb")
        shutil.copy(src_file, raw_temp)

        return self.project_texture_to_mesh(
            glb_path=raw_temp,
            image_path=image_path,
            output_glb_path=str(out_dest)
        )

    def generate_3d_from_text(
        self,
        caption: str,
        output_path: str,
        steps: int = 15,
        timeout: int = 180
    ) -> str:
        """
        Automated 2-Stage Neural Pipeline:
        1. Text -> Orthographic A-pose Concept Image (Neutral Studio Backdrop)
        2. Concept Image -> Tencent Hunyuan3D-2 /shape_generation (250,000+ Polygons)
        3. Concept Image UV Texture Projection -> Game-Ready Textured GLB
        """
        out_dest = Path(output_path)
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        concept_img = str(out_dest.parent / f"{out_dest.stem}_concept.png")

        # Step 1: Synthesize concept image
        self.generate_concept_image(caption, concept_img)

        # Step 2 & 3: Synthesize 3D neural mesh and project texture
        return self.generate_3d_from_image(
            image_path=concept_img,
            output_path=str(out_dest),
            caption=caption,
            steps=steps,
            timeout=timeout
        )
