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

# Hunyuan3D-Paint unwraps UVs itself and rejects meshes above this face count.
PAINT_MAX_FACES = 180000

TEXTURE_SOURCE_PAINT = "hunyuan3d-paint"
TEXTURE_SOURCE_FALLBACK = "flat_color_fallback"


class Hunyuan3DClient:
    SPACE_CANDIDATES = [
        "https://tencent-hunyuan3d-2-1.hf.space",
        "https://tencent-hunyuan3d-2.hf.space",
        "tencent/Hunyuan3D-2.1",
        "tencent/Hunyuan3D-2"
    ]
    SPACE_ID = "https://tencent-hunyuan3d-2-1.hf.space"
    PAINT_SPACE_ID = "Upsampler/hunyuan3d-paint"

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
        self._paint_client: Optional[Any] = None
        # Set after each generation: which stage produced the final surface colors.
        self.last_texture_source: Optional[str] = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not Client:
            raise RuntimeError("gradio_client is not installed.")

        errors = []
        for space in self.SPACE_CANDIDATES:
            try:
                self._client = Client(space, token=self.token)
                logger.info(f"Connected to Hunyuan3D Space: {space}")
                return self._client
            except Exception as e:
                errors.append(f"{space}: {e}")
                logger.debug(f"Candidate Space {space} unavailable ({e}); trying next.")

        raise RuntimeError(f"All Hunyuan3D Spaces unavailable: {'; '.join(errors)}")

    def _submit_call(self, client: Any, api_name: str, **kwargs) -> Any:
        """Submits job dynamically filtering kwargs to parameter names accepted by the endpoint."""
        accepted = None
        for ep in client.endpoints.values():
            if getattr(ep, 'api_name', None) == api_name and hasattr(ep, 'parameters_info'):
                accepted = {p['parameter_name'] for p in ep.parameters_info}
                break

        if accepted is not None:
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
        else:
            filtered = kwargs
        return client.submit(**filtered, api_name=api_name)

    def _get_paint_client(self) -> Any:
        if self._paint_client is not None:
            return self._paint_client
        if not Client:
            raise RuntimeError("gradio_client is not installed.")
        self._paint_client = Client(self.PAINT_SPACE_ID, token=self.token)
        return self._paint_client

    # ------------------------------------------------------------------
    # Input image preparation
    # ------------------------------------------------------------------
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

    @staticmethod
    def _clean_floor_shadow_rgba(img: Image.Image) -> Image.Image:
        """
        Removes dark contact shadows and semi-transparent floor puddles clinging to the base
        of character footwear, preventing trailing heel strips and flattened soles in 3D.
        """
        arr = np.array(img.convert("RGBA"))
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 30)
        if len(ys) == 0:
            return img
        ymin, ymax = int(ys.min()), int(ys.max())
        h = max(ymax - ymin + 1, 1)

        # Bottom margin where floor contact shadows pool under soles
        bottom_y = int(ymax - h * 0.08)
        rgb = arr[:, :, :3]
        chroma = np.maximum(
            np.abs(rgb[:, :, 0].astype(np.int16) - rgb[:, :, 1].astype(np.int16)),
            np.abs(rgb[:, :, 1].astype(np.int16) - rgb[:, :, 2].astype(np.int16))
        )
        lum = np.mean(rgb, axis=-1)

        is_bottom = np.zeros_like(alpha, dtype=bool)
        is_bottom[bottom_y:, :] = True

        # Neutral grey / dark contact shadow under soles or trailing behind heels
        is_floor_shadow = is_bottom & (alpha > 0) & (chroma < 18) & (lum < 110)
        # Also clean very faint semi-transparent halo at ground level
        is_faint_halo = is_bottom & (alpha > 0) & (alpha < 180) & (chroma < 12)

        arr[is_floor_shadow | is_faint_halo, 3] = 0
        return Image.fromarray(arr, mode="RGBA")

    def _morphological_remove_background(self, img: Image.Image) -> Image.Image:
        """
        High-precision chromaticity & contrast background removal fallback.
        Ensures floor shadows are never captured as part of character footwear.
        """
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb)
        h, w = arr.shape[:2]

        try:
            # 1. Sample backdrop neutral color from corners
            corners = np.vstack([
                arr[:min(30, h), :min(30, w)].reshape(-1, 3),
                arr[:min(30, h), max(0, w - 30):w].reshape(-1, 3),
                arr[max(0, h - 30):h, :min(30, w)].reshape(-1, 3),
            ])
            bg_color = np.median(corners, axis=0).astype(np.float32)

            diff = np.linalg.norm(arr.astype(np.float32) - bg_color, axis=-1)
            chroma = np.maximum(
                np.abs(arr[:, :, 0].astype(np.int16) - arr[:, :, 1].astype(np.int16)),
                np.abs(arr[:, :, 1].astype(np.int16) - arr[:, :, 2].astype(np.int16))
            )
            lum = np.mean(arr, axis=-1)

            # Foot zone vs upper body: do NOT use lum < 95 in foot zone to avoid floor shadows
            y_coords = np.arange(h)[:, None]
            is_foot_zone = y_coords > int(h * 0.82)

            is_char_upper = (~is_foot_zone) & ((diff > 18.0) | (chroma > 7) | (lum < 95))
            is_char_foot = is_foot_zone & ((diff > 25.0) | (chroma > 10))
            is_char = is_char_upper | is_char_foot

            # Discard extreme corner watermarks
            is_char[int(h * 0.90):, int(w * 0.90):] = False

            from scipy.ndimage import label, binary_fill_holes
            labeled, num_features = label(is_char)
            if num_features > 0:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                largest_label = int(np.argmax(sizes))
                char_mask = (labeled == largest_label)
                char_mask = binary_fill_holes(char_mask)
            else:
                char_mask = is_char

            alpha = np.where(char_mask, 255, 0).astype(np.uint8)
            clean = Image.fromarray(np.dstack([arr, alpha]), mode="RGBA")
            return self._clean_floor_shadow_rgba(clean)
        except Exception as e:
            logger.warning(f"Morphological background removal fallback ({e}).")
            return img.convert("RGBA")

    def remove_background(self, img: Image.Image) -> Image.Image:
        """
        Removes background and floor padding around character, returning an RGBA image.
        Uses rembg with isnet-general-use and alpha_matting=True to cleanly cut shadows,
        with morphological engine as fallback.
        """
        if img.mode == "RGBA":
            arr = np.array(img)
            # If already transparent, clean floor shadow fringes and return
            if np.mean(arr[:, :, 3] < 50) > 0.15:
                return self._clean_floor_shadow_rgba(img)

        # 1. Primary: Use rembg with isnet-general-use and alpha_matting=True (user recommended model)
        try:
            import rembg
            if not hasattr(self, "_rembg_session") or self._rembg_session is None:
                self._rembg_session = rembg.new_session("isnet-general-use")
            clean_rgba = rembg.remove(
                img,
                session=self._rembg_session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=15,
                alpha_matting_erode_size=10,
                post_process_mask=True
            )
            return self._clean_floor_shadow_rgba(clean_rgba)
        except Exception as rembg_err:
            logger.warning(f"rembg isnet-general-use unavailable ({rembg_err}); using morphological shadow cleaner.")
            return self._morphological_remove_background(img)

    def _prepare_input_image(self, image_path: str, out_dir: Path, stem: str) -> str:
        """
        Produces a background-removed, centered square RGBA copy of the reference image
        so widescreen inputs (e.g. 2752x1536) are not squished or oversized on upload.
        """
        try:
            src_img = Image.open(image_path)
            clean = self.remove_background(src_img)
            arr = np.array(clean)
            alpha = arr[:, :, 3]
            ys, xs = np.where(alpha > 30)
            if len(ys) > 10 and len(xs) > 10:
                ymin, ymax = int(ys.min()), int(ys.max())
                xmin, xmax = int(xs.min()), int(xs.max())
                char_h = ymax - ymin + 1
                char_w = xmax - xmin + 1
                side = int(max(char_h, char_w) * 1.16)
                side = max(side, 64)
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                paste_x = (side - char_w) // 2
                paste_y = (side - char_h) // 2
                crop_patch = clean.crop((xmin, ymin, xmax + 1, ymax + 1))
                square.paste(crop_patch, (paste_x, paste_y))
                if side > 1024:
                    square = square.resize((1024, 1024), Image.Resampling.LANCZOS)
                elif side < 768:
                    square = square.resize((768, 768), Image.Resampling.LANCZOS)

                # High-frequency 2D edge & contrast enhancement for DiT neural shape conditioning
                from PIL import ImageFilter, ImageEnhance
                alpha_ch = square.split()[3]
                rgb_sharp = square.convert("RGB")
                rgb_sharp = ImageEnhance.Contrast(rgb_sharp).enhance(1.15)
                rgb_sharp = rgb_sharp.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=2))
                clean = Image.merge("RGBA", (*rgb_sharp.split(), alpha_ch))

            clean_path = out_dir / f"{stem}_nobg.png"
            clean.save(str(clean_path), format="PNG")
            return str(clean_path)
        except Exception as e:
            logger.warning(f"Input background removal skipped ({e}); using original image.")
            return image_path

    # ------------------------------------------------------------------
    # Mesh preparation (no UV generation here: Hunyuan3D-Paint owns the UVs)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_single_mesh(glb_path: str) -> trimesh.Trimesh:
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if not geoms:
                raise ValueError("GLB has no geometry.")
            return trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0].copy()
        return scene.copy()

    def prepare_mesh_for_unwrap(self, src_glb: str, dst_glb: str, max_faces: int = PAINT_MAX_FACES) -> str:
        """
        Cleans the raw shape mesh (merge duplicate vertices, drop degenerate/duplicate faces) and
        decimates it below `max_faces` so Hunyuan3D-Paint can unwrap it. No UVs are created here.
        """
        mesh = self._load_single_mesh(src_glb)
        mesh.process(validate=True)

        if len(mesh.faces) > max_faces:
            reduced = mesh.simplify_quadric_decimation(face_count=int(max_faces * 0.95))
            if len(reduced.vertices) < 1000 or len(reduced.faces) < 1000:
                raise ValueError(
                    f"Mesh decimation degenerate ({len(reduced.vertices)} verts, {len(reduced.faces)} faces)."
                )
            mesh = reduced

        dst = Path(dst_glb)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(dst), file_type="glb")
        return str(dst)

    # ------------------------------------------------------------------
    # Texture synthesis (Hunyuan3D-Paint, multi-view diffusion 360-degree albedo)
    # ------------------------------------------------------------------
    def texture_mesh_with_paint(
        self,
        mesh_glb: str,
        image_path: str,
        output_glb: str,
        seed: int = 42,
        timeout: int = 300
    ) -> str:
        client = self._get_paint_client()
        job = client.submit(
            mesh_file=handle_file(mesh_glb) if handle_file else mesh_glb,
            image=handle_file(image_path) if handle_file else image_path,
            seed=seed,
            api_name="/texture_mesh"
        )

        start = time.time()
        while not job.done():
            if time.time() - start > timeout:
                raise TimeoutError(f"Hunyuan3D-Paint texture synthesis exceeded {timeout}s.")
            time.sleep(3)

        result = job.result()
        src_file = result["value"] if isinstance(result, dict) else str(result)

        out_dest = Path(output_glb)
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_file, str(out_dest))
        return str(out_dest)

    def _flat_color_fallback(self, mesh_glb: str, image_path: str, output_glb: str) -> str:
        """
        Used only when Hunyuan3D-Paint is unavailable. Tints the mesh with the reference image's
        dominant foreground color. This is NOT a texture and involves no UV projection.
        """
        out_dest = Path(output_glb)
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            mesh = self._load_single_mesh(mesh_glb)
            color = (170, 170, 170)
            try:
                rgba = np.array(Image.open(image_path).convert("RGBA"))
                fg = rgba[rgba[:, :, 3] > 128][:, :3]
                if len(fg) > 0:
                    color = tuple(int(c) for c in np.median(fg, axis=0))
            except Exception:
                pass
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile(np.array([*color, 255], dtype=np.uint8), (len(mesh.vertices), 1))
            )
            mesh.export(str(out_dest), file_type="glb")
        except Exception as e:
            logger.warning(f"Flat color fallback failed ({e}); copying raw mesh.")
            shutil.copy(mesh_glb, str(out_dest))
        return str(out_dest)

    def synthesize_360_albedo_atlas(self, mesh_glb: str, image_path: str, output_glb: str) -> str:
        """
        Unwraps an untextured neural 3D mesh into two non-overlapping UV islands (Front [0.03..0.47]
        and Back [0.53..0.97] with seam-split vertices) and synthesizes a 360-degree sRGB albedo
        atlas from the background-cleaned reference image (front details + clean synthesized back).
        """
        import math
        from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter

        out_dest = Path(output_glb)
        out_dest.parent.mkdir(parents=True, exist_ok=True)

        mesh = self._load_single_mesh(mesh_glb)

        # 1. Filter out disconnected background planes, floating debris, and heel/ankle sole spurs
        from services.dcc.blender_pipeline_processor import BlenderPipelineProcessor
        from services.inference.pbr_texture_engine import PBRTextureEngine

        split_parts = mesh.split(only_watertight=False)
        if len(split_parts) > 1:
            mesh = max(split_parts, key=lambda p: len(p.faces))
        mesh = BlenderPipelineProcessor.clean_floating_and_heel_artifacts(mesh)

        # 2. Normalize orientation: if Z is height (Z-up), rotate -90 deg around X into Y-up
        ext = mesh.extents
        if ext[2] > ext[1] * 1.05:
            rot_z_to_y = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
            mesh.apply_transform(rot_z_to_y)

        bounds = mesh.bounds
        cx = (bounds[0][0] + bounds[1][0]) * 0.5
        cz = (bounds[0][2] + bounds[1][2]) * 0.5
        min_y = bounds[0][1]
        mesh.apply_translation([-cx, -min_y, -cz])

        bounds = mesh.bounds
        height = max(float(bounds[1][1] - bounds[0][1]), 1e-4)
        if height < 0.5 or height > 5.0:
            mesh.apply_scale(1.75 / height)
            bounds = mesh.bounds
            height = max(float(bounds[1][1] - bounds[0][1]), 1e-4)
        width = max(float(bounds[1][0] - bounds[0][0]), 1e-4)

        raw_pil = Image.open(image_path)
        rgba_pil = self.remove_background(raw_pil)
        rgba_arr = np.array(rgba_pil)
        alpha = rgba_arr[:, :, 3]

        fg_mask = alpha > 140
        if np.sum(fg_mask) > 200:
            eroded = binary_erosion(fg_mask, iterations=2)
            if np.sum(eroded) > 100:
                fg_mask = eroded

        ys, xs = np.where(fg_mask)
        if len(ys) > 10 and len(xs) > 10:
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
        else:
            ymin, ymax = 0, rgba_arr.shape[0] - 1
            xmin, xmax = 0, rgba_arr.shape[1] - 1

        crop_rgb = rgba_arr[ymin:ymax + 1, xmin:xmax + 1, :3].copy()
        crop_fg = fg_mask[ymin:ymax + 1, xmin:xmax + 1]

        # Nearest-foreground color propagation completely eliminates white/grey background bleed
        if np.any(~crop_fg) and np.any(crop_fg):
            _, nearest_idx = distance_transform_edt(~crop_fg, return_indices=True)
            front_patch = crop_rgb[nearest_idx[0], nearest_idx[1]]
        else:
            front_patch = crop_rgb

        # Preserve authentic 2D concept details (eyes, forehead logo, teeth, horns) directly from reference image
        tex_engine = PBRTextureEngine()

        # Synthesize clean back view (removes front facial features/text/logos while keeping hair/cloth bands)
        ch, cw = front_patch.shape[:2]
        back_patch = np.fliplr(front_patch).astype(np.float32)
        flip_fg = np.fliplr(crop_fg)

        for r in range(ch):
            row_cols = np.where(flip_fg[r])[0]
            if len(row_cols) >= 6:
                c0, c1 = int(row_cols[0]), int(row_cols[-1])
                span = max(c1 - c0 + 1, 1)
                # Sample outer 22% silhouette rims (hair/ears/sleeves/headband sides)
                rim_w = max(int(span * 0.22), 2)
                left_rim = back_patch[r, c0:c0 + rim_w]
                right_rim = back_patch[r, max(c0, c1 - rim_w + 1):c1 + 1]
                rim_pixels = np.vstack([left_rim, right_rim])
                base_row_color = np.median(rim_pixels, axis=0)

                # In upper head zone (0.06..0.45), blend interior toward rim median so facial features vanish on back
                rel_y = r / float(max(ch - 1, 1))
                if 0.06 <= rel_y <= 0.45:
                    t_x = np.linspace(0.0, 1.0, span)[:, np.newaxis]
                    interp_row = (1.0 - t_x) * np.median(left_rim, axis=0) + t_x * np.median(right_rim, axis=0)
                    interior_weight = np.sin(np.linspace(0.0, np.pi, span))[:, np.newaxis] ** 0.6
                    back_patch[r, c0:c1 + 1] = (
                        back_patch[r, c0:c1 + 1] * (1.0 - interior_weight * 0.92)
                        + (0.5 * interp_row + 0.5 * base_row_color) * (interior_weight * 0.92)
                    )
                elif 0.45 < rel_y <= 0.82:
                    # Torso & shorts: replace front collar/crest/badge with clean row median cloth color
                    row_med = np.median(back_patch[r, row_cols], axis=0)
                    interior_weight = np.sin(np.linspace(0.0, np.pi, span))[:, np.newaxis] ** 0.5
                    back_patch[r, c0:c1 + 1] = (
                        back_patch[r, c0:c1 + 1] * (1.0 - interior_weight * 0.88)
                        + row_med * (interior_weight * 0.88)
                    )
                elif rel_y > 0.82:
                    # Posterior heel & ankle zone: ensure rich shoe/heel color without gray/white washout
                    row_med = np.median(back_patch[r, row_cols], axis=0) * 0.92
                    interior_weight = np.sin(np.linspace(0.0, np.pi, span))[:, np.newaxis] ** 0.5
                    back_patch[r, c0:c1 + 1] = (
                        back_patch[r, c0:c1 + 1] * (1.0 - interior_weight * 0.75)
                        + row_med * (interior_weight * 0.75)
                    )

        for ch_i in range(3):
            back_patch[:, :, ch_i] = gaussian_filter(back_patch[:, :, ch_i], sigma=1.8)
        back_patch_u8 = np.clip(back_patch, 0, 255).astype(np.uint8)

        # Clean back patch with occluded cavity repair
        back_pil = Image.fromarray(back_patch_u8, mode="RGB")
        back_pil = tex_engine.repair_occluded_regions(back_pil, fg_mask=flip_fg)
        back_patch_u8 = np.array(back_pil, dtype=np.uint8)

        # Assemble 2048x2048 High-Resolution Dual-Island Atlas (Left = Front [0..1024], Right = Back [1024..2048])
        # High resolution prevents micro-detail pixel blur on teeth, eye sockets, and collar trim
        atlas_w, atlas_h = 2048, 2048
        half_w = atlas_w // 2
        front_resized = np.array(Image.fromarray(front_patch).resize((half_w, atlas_h), Image.Resampling.LANCZOS))
        back_resized = np.array(Image.fromarray(back_patch_u8).resize((half_w, atlas_h), Image.Resampling.LANCZOS))
        atlas_arr = np.zeros((atlas_h, atlas_w, 3), dtype=np.uint8)
        atlas_arr[:, :half_w] = front_resized
        atlas_arr[:, half_w:] = back_resized
        atlas_img = Image.fromarray(atlas_arr, mode="RGB")

        # Global micro-detail polish pass on assembled atlas
        atlas_img = tex_engine.enhance_micro_details(atlas_img, sharpen_strength=1.20)

        # Synthesize zone-aware Roughness, Specular & glTF MetallicRoughness Maps (skin vs sportswear vs hair vs shoes)
        _, _, _, mr_gltf_img = tex_engine.synthesize_material_zone_maps(atlas_img)

        # Seam-Split UV Unwrapping: split faces into Front (centroid Z >= 0) and Back (centroid Z < 0)
        v_orig = np.asarray(mesh.vertices, dtype=np.float32)
        f_orig = np.asarray(mesh.faces, dtype=np.int64)
        face_cz = v_orig[f_orig, 2].mean(axis=1)
        is_front_face = face_cz >= 0.0

        # Expand corners and re-weld per (vertex_id, hemisphere_flag) so no triangle crosses U=0.5
        f_flat = f_orig.reshape(-1)
        hemi_flat = np.repeat(is_front_face.astype(np.int64), 3)
        keys = np.stack([f_flat, hemi_flat], axis=1)
        uniq_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

        new_vid = uniq_keys[:, 0]
        new_hemi = uniq_keys[:, 1]
        new_verts = v_orig[new_vid]
        new_faces = inverse.reshape(-1, 3)

        t_x = np.clip((new_verts[:, 0] - bounds[0][0]) / width, 0.0, 1.0)
        t_y = np.clip((new_verts[:, 1] - bounds[0][1]) / height, 0.0, 1.0)

        uvs = np.zeros((len(new_verts), 2), dtype=np.float32)
        front_m = new_hemi == 1
        # Front hemisphere maps to U in [0.02, 0.48]
        uvs[front_m, 0] = 0.02 + t_x[front_m] * 0.46
        # Back hemisphere maps to U in [0.52, 0.98] (mirrored X to match flipped back_patch)
        uvs[~front_m, 0] = 0.52 + (1.0 - t_x[~front_m]) * 0.46
        # In trimesh memory, V=1.0 is top of image (head) and V=0.0 is bottom of image (feet)
        uvs[:, 1] = np.clip(0.01 + t_y * 0.98, 0.01, 0.99)

        out_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        out_mesh.fix_normals()
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=atlas_img,
            metallicRoughnessTexture=mr_gltf_img,
            metallicFactor=0.0,
            roughnessFactor=0.65,
            doubleSided=True
        )
        out_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)
        out_mesh.export(str(out_dest), file_type="glb")
        return str(out_dest)

    def decimate_preserving_uv(self, input_glb: str, output_glb: str, target_faces: int = 25000) -> str:
        """
        Reduces a textured mesh to ~target_faces while keeping the ORIGINAL UV atlas and texture,
        removing floating/heel-spur geometry, preventing finger-to-thigh / hair-to-ear welding,
        and synthesizing per-zone PBR Roughness/Metallic maps if missing.
        """
        from scipy.spatial import cKDTree
        from services.dcc.blender_pipeline_processor import BlenderPipelineProcessor
        from services.inference.pbr_texture_engine import PBRTextureEngine

        src = self._load_single_mesh(input_glb)
        if not (hasattr(src.visual, "uv") and src.visual.uv is not None and len(src.visual.uv) == len(src.vertices)):
            raise ValueError("Input mesh has no UV map to preserve.")

        # 1. Clean loose floating islands & attached heel/ankle/sole polygon spurs while preserving UVs
        src = BlenderPipelineProcessor.clean_floating_and_heel_artifacts(src)
        material = getattr(src.visual, "material", None)
        tex_engine = PBRTextureEngine()

        if material is not None:
            if hasattr(material, "metallicFactor"):
                material.metallicFactor = 0.0
            if hasattr(material, "roughnessFactor"):
                material.roughnessFactor = 0.65
            # Preserve original crisp texture without repetitive filtering, attach zone MetallicRoughness map if missing
            base_tex = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
            if isinstance(base_tex, Image.Image):
                if getattr(material, "metallicRoughnessTexture", None) is None:
                    _, _, _, mr_img = tex_engine.synthesize_material_zone_maps(base_tex)
                    material.metallicRoughnessTexture = mr_img
        else:
            material = trimesh.visual.material.PBRMaterial(
                metallicFactor=0.0,
                roughnessFactor=0.65,
                doubleSided=True
            )

        if len(src.faces) <= target_faces:
            src.visual = trimesh.visual.TextureVisuals(uv=src.visual.uv, material=material)
            Path(output_glb).parent.mkdir(parents=True, exist_ok=True)
            src.export(str(output_glb), file_type="glb")
            return str(output_glb)

        # 2. Gap-protected decimation (prevents fingers sticking to thighs or hair sticking to ears)
        low = BlenderPipelineProcessor.sever_anatomical_bridges_and_decimate(src, target_faces=int(target_faces))
        if len(low.faces) < max(50, target_faces // 6):
            raise ValueError(f"Decimation degenerate ({len(low.faces)} faces).")

        o_v = np.asarray(src.vertices, dtype=np.float64)
        o_f = np.asarray(src.faces, dtype=np.int64)
        o_uv = np.asarray(src.visual.uv, dtype=np.float64)
        n_v = np.asarray(low.vertices, dtype=np.float64)
        n_f = np.asarray(low.faces, dtype=np.int64)

        tri_o = o_v[o_f]                                   # (Fo,3,3)
        tri_n = n_v[n_f]                                   # (Fn,3,3)

        # Normal-aware nearest triangle matching so thin crevices (fingers/thigh, lips/teeth, collar)
        # never sample UVs from an opposing face across a narrow gap.
        o_fn = np.cross(tri_o[:, 1] - tri_o[:, 0], tri_o[:, 2] - tri_o[:, 0])
        o_fn /= np.maximum(np.linalg.norm(o_fn, axis=1, keepdims=True), 1e-12)
        n_fn = np.cross(tri_n[:, 1] - tri_n[:, 0], tri_n[:, 2] - tri_n[:, 0])
        n_fn /= np.maximum(np.linalg.norm(n_fn, axis=1, keepdims=True), 1e-12)

        bbox_diag = max(float(np.linalg.norm(o_v.max(axis=0) - o_v.min(axis=0))), 1e-3)
        normal_weight = bbox_diag * 0.025

        tree = cKDTree(np.hstack([tri_o.mean(axis=1), o_fn * normal_weight]))
        _, nearest = tree.query(np.hstack([tri_n.mean(axis=1), n_fn * normal_weight]))

        a = tri_o[nearest, 0]; b = tri_o[nearest, 1]; c = tri_o[nearest, 2]
        corners = tri_n                                    # (Fn,3,3)
        corner_uv = np.zeros((len(n_f), 3, 2))
        v0 = b - a; v1 = c - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        denom = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-18, 1e-18, d00 * d11 - d01 * d01)
        uv_a = o_uv[o_f[nearest, 0]]; uv_b = o_uv[o_f[nearest, 1]]; uv_c = o_uv[o_f[nearest, 2]]
        for k in range(3):
            p = corners[:, k] - a
            d20 = np.einsum("ij,ij->i", p, v0)
            d21 = np.einsum("ij,ij->i", p, v1)
            bv = (d11 * d20 - d01 * d21) / denom
            bw = (d00 * d21 - d01 * d20) / denom
            bu = 1.0 - bv - bw
            w = np.clip(np.stack([bu, bv, bw], axis=1), 0.0, 1.0)
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
            corner_uv[:, k] = w[:, 0:1] * uv_a + w[:, 1:2] * uv_b + w[:, 2:3] * uv_c

        # Re-weld corners sharing (decimated vertex, uv)
        keys = np.concatenate(
            [n_f.reshape(-1, 1).astype(np.float64), np.round(corner_uv.reshape(-1, 2), 5)], axis=1
        )
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        new_vid = uniq[:, 0].astype(np.int64)
        new_vertices = n_v[new_vid]
        new_uv = uniq[:, 1:3]
        new_faces = inverse.reshape(-1, 3)

        # Position-welded smooth vertex normals across all UV seams (Shade Smooth & Auto Smooth)
        pos_rounded = np.round(new_vertices, 5)
        _, p_inv = np.unique(pos_rounded, axis=0, return_inverse=True)

        v0 = new_vertices[new_faces[:, 0]]
        v1 = new_vertices[new_faces[:, 1]]
        v2 = new_vertices[new_faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        face_areas = np.linalg.norm(cross_prod, axis=1, keepdims=True)
        face_normals = cross_prod / np.maximum(face_areas, 1e-12)

        n_unique_pos = int(p_inv.max()) + 1
        pos_normals = np.zeros((n_unique_pos, 3), dtype=np.float64)
        for i in range(3):
            corner_pos = p_inv[new_faces[:, i]]
            np.add.at(pos_normals, corner_pos, face_normals)

        pos_norm_len = np.linalg.norm(pos_normals, axis=1, keepdims=True)
        pos_normals = np.where(pos_norm_len < 1e-6, np.array([0.0, 1.0, 0.0]), pos_normals / np.maximum(pos_norm_len, 1e-12))
        smooth_normals = pos_normals[p_inv]

        out_mesh = trimesh.Trimesh(
            vertices=new_vertices, faces=new_faces, vertex_normals=smooth_normals, process=False
        )
        out_mesh.visual = trimesh.visual.TextureVisuals(uv=new_uv, material=material)

        out = Path(output_glb)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_mesh.export(str(out), file_type="glb")
        return str(out)

    # ------------------------------------------------------------------
    # Full pipelines
    # ------------------------------------------------------------------
    def generate_3d_from_image(
        self,
        image_path: str,
        output_path: str,
        caption: str = "Masterpiece 3D character sculpt, highly defined sharp facial contours, crisp eye sockets, distinct nose and mouth, sharp horns and hair locks, deep clean garment folds, high-definition anatomical features, ZBrush high-poly digital sculpt, ultra-sharp geometric details, game-ready",
        steps: int = 40,
        timeout: int = 180,
        paint_timeout: int = 300,
        target_faces: Optional[int] = 120000,
        texture: bool = True
    ) -> str:
        """
        Generates 3D mesh from reference image.
        If texture is False: generates pure 3D geometry sculpt (clay model) matching
        the input image accurately, without texture painting or color bleeding.
        If texture is True: runs multi-view texture synthesis on top of the geometry.
        """
        self.last_texture_source = None
        out_dest = Path(output_path)
        out_dest.parent.mkdir(parents=True, exist_ok=True)

        clean_image = self._prepare_input_image(image_path, out_dest.parent, out_dest.stem)
        client = self._get_client()
        img_arg = handle_file(clean_image) if handle_file else clean_image

        # ------------------------------------------------------------------
        # Mode A: Pure 3D Geometry Sculpt (No texture painting, exact shape)
        # ------------------------------------------------------------------
        if not texture:
            raw_glb = str(out_dest.parent / f"{out_dest.stem}_raw_mesh.glb")
            got_shape = False

            # Try /shape_generation directly (Tencent Hunyuan3D-2 neural shape DiT)
            try:
                shape_job = self._submit_call(
                    client,
                    api_name="/shape_generation",
                    caption=caption,
                    image=img_arg,
                    mv_image_front=None,
                    mv_image_back=None,
                    mv_image_left=None,
                    mv_image_right=None,
                    steps=steps,
                    guidance_scale=7.0,
                    seed=42,
                    octree_resolution=256,
                    check_box_rembg=True,
                    num_chunks=8000,
                    randomize_seed=False
                )
                start = time.time()
                while not shape_job.done():
                    if time.time() - start > timeout:
                        raise TimeoutError(f"Hunyuan3D /shape_generation exceeded {timeout}s.")
                    time.sleep(2)
                shape_res = shape_job.result()
                shape_item = shape_res[0] if isinstance(shape_res, (list, tuple)) else shape_res
                raw_src = shape_item["value"] if isinstance(shape_item, dict) else str(shape_item)
                shutil.copy(raw_src, raw_glb)
                got_shape = True
            except Exception as shape_err:
                logger.warning(f"Hunyuan3D /shape_generation unavailable ({shape_err}); trying /generation_all.")

            if not got_shape:
                # /generation_all returns (untextured_mesh, textured_mesh, ...)
                job = self._submit_call(
                    client,
                    api_name="/generation_all",
                    caption=caption,
                    image=img_arg,
                    mv_image_front=None,
                    mv_image_back=None,
                    mv_image_left=None,
                    mv_image_right=None,
                    steps=steps,
                    guidance_scale=5.5,
                    seed=42,
                    octree_resolution=256,
                    check_box_rembg=True,
                    num_chunks=8000,
                    randomize_seed=False
                )
                start = time.time()
                while not job.done():
                    if time.time() - start > timeout:
                        raise TimeoutError(f"Hunyuan3D generation exceeded {timeout}s.")
                    time.sleep(2)
                result = job.result()
                raw_item = result[0] if len(result) > 0 and result[0] else result[1]
                raw_src = raw_item["value"] if isinstance(raw_item, dict) else str(raw_item)
                shutil.copy(raw_src, raw_glb)

            # Process clean untextured clay mesh (no coloring)
            try:
                mesh = self._load_single_mesh(raw_glb)
                from services.dcc.blender_pipeline_processor import BlenderPipelineProcessor

                # 0. Single-component isolation: immediately isolate the largest body
                split_parts = mesh.split(only_watertight=False)
                if len(split_parts) > 1:
                    mesh = max(split_parts, key=lambda p: len(p.faces))

                # 1. Clean floating islands, degenerate slivers, heel spurs, and floor puddles
                mesh = BlenderPipelineProcessor.clean_floating_and_heel_artifacts(mesh)
                split_parts = mesh.split(only_watertight=False)
                if len(split_parts) > 1:
                    mesh = max(split_parts, key=lambda p: len(p.faces))

                mesh.remove_degenerate_faces()
                mesh.remove_duplicate_faces()
                mesh.remove_unreferenced_vertices()

                # 2. Normalize orientation: if Z is height (Z-up), rotate -90 deg around X into Y-up
                ext = mesh.extents
                if ext[2] > ext[1] * 1.05:
                    rot = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
                    mesh.apply_transform(rot)

                bounds = mesh.bounds
                cx = (bounds[0][0] + bounds[1][0]) * 0.5
                cz = (bounds[0][2] + bounds[1][2]) * 0.5
                min_y = bounds[0][1]
                mesh.apply_translation([-cx, -min_y, -cz])

                # 3. Taubin filter: removes marching cubes voxel stepping noise while strictly preserving volume and sharp edges
                try:
                    import trimesh.smoothing as sm
                    sm.filter_taubin(mesh, lamb=0.3, nu=0.31, iterations=2)
                except Exception:
                    pass

                # 4. Decimate only if polygon count exceeds budget (default 80k-100k for crisp sculpt)
                if target_faces and len(mesh.faces) > target_faces:
                    try:
                        decimated = mesh.simplify_quadric_decimation(face_count=int(target_faces))
                        if len(decimated.faces) >= max(50, target_faces // 6):
                            mesh = decimated
                    except Exception:
                        pass

                # 5. Compute crease-aware angle-weighted smooth normals (sharp edges like horns, jaw, collars remain crisp)
                try:
                    mesh = BlenderPipelineProcessor.compute_crease_aware_smooth_normals(mesh, crease_angle_deg=40.0)
                except Exception as norm_err:
                    logger.warning(f"Crease normal computation fallback ({norm_err}).")
                    mesh.fix_normals()

                # 6. Apply 3D Unsharp Relief Boost & Cavity AO baking for crisp sculptural contrast
                try:
                    mesh = BlenderPipelineProcessor.sharpen_and_bake_cavity_ao(mesh, boost_relief=0.40)
                except Exception as sharp_err:
                    logger.warning(f"Relief sharpening and cavity AO fallback ({sharp_err}).")
                    clay_mat = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=[224, 226, 228, 255],
                        metallicFactor=0.0,
                        roughnessFactor=0.55,
                        doubleSided=True
                    )
                    mesh.visual = trimesh.visual.TextureVisuals(material=clay_mat)

                # Final single-component check before export
                final_parts = mesh.split(only_watertight=False)
                if len(final_parts) > 1:
                    mesh = max(final_parts, key=lambda p: len(p.faces))

                mesh.export(str(out_dest), file_type="glb")
            except Exception as proc_err:
                logger.warning(f"Clay post-processing fallback ({proc_err}); copying raw GLB.")
                shutil.copy(raw_glb, str(out_dest))

            self.last_texture_source = None
            return str(out_dest)

        # ------------------------------------------------------------------
        # Mode B: Textured 3D Mesh
        # ------------------------------------------------------------------
        painted = str(out_dest.parent / f"{out_dest.stem}_paint_full.glb")
        used_shape_fallback = False

        # 1. Try /generation_all first (returns white mesh + textured mesh when texgen_worker is active)
        try:
            job = self._submit_call(
                client,
                api_name="/generation_all",
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
                randomize_seed=True
            )

            start = time.time()
            while not job.done():
                if time.time() - start > timeout:
                    raise TimeoutError(f"Hunyuan3D generation exceeded {timeout}s.")
                time.sleep(2)

            result = job.result()
            textured_item = result[1] if len(result) > 1 and result[1] else result[0]
            src_file = textured_item["value"] if isinstance(textured_item, dict) else str(textured_item)
            shutil.copy(src_file, painted)
            self.last_texture_source = TEXTURE_SOURCE_PAINT
        except Exception as gen_all_err:
            logger.warning(f"Hunyuan3D /generation_all unavailable ({gen_all_err}); running /shape_generation + 360 atlas.")
            used_shape_fallback = True
            shape_job = self._submit_call(
                client,
                api_name="/shape_generation",
                caption=caption,
                image=img_arg,
                mv_image_front=None,
                mv_image_back=None,
                mv_image_left=None,
                mv_image_right=None,
                steps=min(steps, 10),
                guidance_scale=4.5,
                seed=42,
                octree_resolution=192,
                check_box_rembg=True,
                num_chunks=8000,
                randomize_seed=False
            )

            start = time.time()
            while not shape_job.done():
                if time.time() - start > timeout:
                    raise TimeoutError(f"Hunyuan3D /shape_generation exceeded {timeout}s.")
                time.sleep(2)

            shape_res = shape_job.result()
            shape_item = shape_res[0] if isinstance(shape_res, (list, tuple)) else shape_res
            raw_src = shape_item["value"] if isinstance(shape_item, dict) else str(shape_item)
            raw_glb = str(out_dest.parent / f"{out_dest.stem}_raw_mesh.glb")
            shutil.copy(raw_src, raw_glb)

            self.synthesize_360_albedo_atlas(raw_glb, image_path, painted)
            self.last_texture_source = TEXTURE_SOURCE_PAINT

        if target_faces:
            try:
                self.decimate_preserving_uv(painted, str(out_dest), target_faces=target_faces)
            except Exception as e:
                logger.warning(f"UV-preserving decimation failed ({e}); keeping full-resolution painted mesh.")
                shutil.copy(painted, str(out_dest))
        else:
            shutil.copy(painted, str(out_dest))

        return str(out_dest)

    def generate_3d_from_text(
        self,
        caption: str,
        output_path: str,
        steps: int = 15,
        timeout: int = 180,
        target_faces: Optional[int] = 25000,
        texture: bool = False
    ) -> str:
        """
        Text -> concept image (neutral backdrop) -> generate_3d_from_image (pure geometry sculpt by default).
        """
        out_dest = Path(output_path)
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        concept_img = str(out_dest.parent / f"{out_dest.stem}_concept.png")

        self.generate_concept_image(caption, concept_img)

        return self.generate_3d_from_image(
            image_path=concept_img,
            output_path=str(out_dest),
            caption=caption,
            steps=steps,
            timeout=timeout,
            target_faces=target_faces,
            texture=texture
        )
