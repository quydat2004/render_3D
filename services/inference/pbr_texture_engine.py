import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import sobel, gaussian_filter

def _hex_to_rgb(hex_str: Optional[str], default: Tuple[int, int, int] = (200, 200, 200)) -> Tuple[int, int, int]:
    if not hex_str:
        return default
    try:
        h = str(hex_str).strip().lstrip("#")
        if len(h) == 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        elif len(h) == 3:
            return (int(h[0]*2, 16), int(h[1]*2, 16), int(h[2]*2, 16))
    except Exception:
        pass
    return default

@dataclass
class PBRTextureSet:
    albedo: Image.Image
    normal: Image.Image
    roughness: Image.Image
    alpha_hair: Image.Image
    alpha_cloth: Image.Image

class PBRTextureEngine:
    def __init__(self):
        pass

    def height_to_normal_map(self, height_array: np.ndarray, strength: float = 2.0) -> Image.Image:
        h = height_array.astype(np.float32)
        h_min, h_max = np.min(h), np.max(h)
        if h_max > h_min:
            h = (h - h_min) / (h_max - h_min)
        else:
            h = np.zeros_like(h)

        dx = sobel(h, axis=1) * strength
        dy = sobel(h, axis=0) * strength
        dz = np.ones_like(h, dtype=np.float32)

        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        norm = np.maximum(norm, 1e-6)

        nx = (-dx / norm) * 0.5 + 0.5
        ny = (-dy / norm) * 0.5 + 0.5
        nz = (dz / norm) * 0.5 + 0.5

        rgb = np.stack([
            np.clip(nx * 255.0, 0, 255).astype(np.uint8),
            np.clip(ny * 255.0, 0, 255).astype(np.uint8),
            np.clip(nz * 255.0, 0, 255).astype(np.uint8)
        ], axis=-1)

        return Image.fromarray(rgb, mode="RGB")

    def generate_hair_alpha_texture(
        self,
        width: int = 512,
        height: int = 1024,
        hair_rgb: Tuple[int, int, int] = (30, 30, 36)
    ) -> Image.Image:
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        rng = np.random.RandomState(42)
        num_strands = 130

        # Base and highlight strand shades
        base_r, base_g, base_b = hair_rgb
        for _ in range(num_strands):
            start_x = rng.uniform(width * 0.05, width * 0.95)
            strand_w = rng.uniform(2.0, 7.0)
            alpha_val = rng.randint(220, 256)

            # Subtle tonal variation per strand for depth
            tint = rng.uniform(0.85, 1.15)
            sr = int(np.clip(base_r * tint, 0, 255))
            sg = int(np.clip(base_g * tint, 0, 255))
            sb = int(np.clip(base_b * tint, 0, 255))

            points = []
            curr_x = start_x
            sway = rng.uniform(-15.0, 15.0)
            steps = 24
            for s in range(steps):
                t = s / float(steps - 1)
                y_coord = t * height
                curr_x += sway * math.sin(t * math.pi * 2.0) + rng.uniform(-1.0, 1.0)
                points.append((curr_x, y_coord))

            for i in range(len(points) - 1):
                draw.line(
                    [points[i], points[i+1]],
                    fill=(sr, sg, sb, alpha_val),
                    width=int(strand_w)
                )

        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        return img

    def generate_torn_cloth_alpha(
        self,
        width: int = 1024,
        height: int = 1024,
        cloth_rgb: Tuple[int, int, int] = (245, 248, 252),
        is_tattered: bool = True
    ) -> Image.Image:
        cr, cg, cb = cloth_rgb
        arr = np.full((height, width, 4), [cr, cg, cb, 255], dtype=np.uint8)

        if is_tattered:
            hem_start = int(height * 0.65)
            x_coords = np.arange(width)

            wave = (
                np.sin(x_coords * 0.04) * 25.0 +
                np.sin(x_coords * 0.12) * 15.0 +
                np.sin(x_coords * 0.35) * 8.0 +
                np.sin(x_coords * 0.8) * 4.0
            )
            edge_y = (height * 0.88 + wave).astype(np.int32)

            for col in range(width):
                cutoff = edge_y[col]
                if cutoff < height:
                    arr[cutoff:, col, 3] = 0

            rng = np.random.RandomState(1337)
            for _ in range(35):
                cx = rng.randint(20, width - 20)
                cy = rng.randint(hem_start, height - 30)
                rx = rng.randint(4, 18)
                ry = rng.randint(8, 35)

                y_grid, x_grid = np.ogrid[
                    max(0, cy - ry):min(height, cy + ry),
                    max(0, cx - rx):min(width, cx + rx)
                ]
                dist = ((x_grid - cx) / rx)**2 + ((y_grid - cy) / ry)**2
                mask = dist <= 1.0
                arr[max(0, cy - ry):min(height, cy + ry), max(0, cx - rx):min(width, cx + rx), 3][mask] = 0

        img = Image.fromarray(arr, mode="RGBA")
        return img

    def generate_character_textures(
        self,
        archetype_name: str,
        width: int = 1024,
        height: int = 1024,
        palette: Optional[Any] = None
    ) -> PBRTextureSet:
        is_ghost = ("ghost" in str(archetype_name).lower() or (palette and getattr(palette, "clothing_type", None) == "dress" and "ghost" in getattr(palette, "skin_color_hex", "")))

        # 1. Resolve Dynamic Palette
        if palette is not None:
            skin_rgb = _hex_to_rgb(getattr(palette, "skin_color_hex", None), (232, 192, 168))
            hair_rgb = _hex_to_rgb(getattr(palette, "hair_color_hex", None), (30, 30, 36))
            eye_rgb = _hex_to_rgb(getattr(palette, "eye_color_hex", None), (0, 240, 255))
            cloth_rgb = _hex_to_rgb(getattr(palette, "primary_cloth_hex", None), (42, 50, 64))
            trim_rgb = _hex_to_rgb(getattr(palette, "secondary_cloth_hex", None), (24, 28, 36))
            glow_hex = getattr(palette, "accent_glow_hex", None)
            glow_rgb = _hex_to_rgb(glow_hex) if glow_hex else None
            mat_metal = float(getattr(palette, "metalness", 0.1))
            mat_rough = float(getattr(palette, "roughness", 0.6))
            is_tattered = is_ghost
        else:
            if is_ghost:
                skin_rgb = (238, 242, 246)
                hair_rgb = (15, 16, 22)
                eye_rgb = (210, 20, 25)
                cloth_rgb = (245, 248, 252)
                trim_rgb = (125, 12, 20)
                glow_rgb = None
                mat_metal = 0.05
                mat_rough = 0.80
                is_tattered = True
            else:
                skin_rgb = (232, 192, 168)
                hair_rgb = (35, 28, 24)
                eye_rgb = (40, 80, 180)
                cloth_rgb = (42, 50, 64)
                trim_rgb = (20, 24, 32)
                glow_rgb = (0, 240, 255)
                mat_metal = 0.40
                mat_rough = 0.55
                is_tattered = False

        base_skin = np.array(skin_rgb, dtype=np.float32)
        base_cloth = np.array(cloth_rgb, dtype=np.float32)
        base_trim = np.array(trim_rgb, dtype=np.float32)

        albedo_arr = np.zeros((height, width, 3), dtype=np.float32)

        # Upper region (Head/Face/Hands): Skin tone
        # Lower region: Primary outfit
        y_split = int(height * 0.36)
        albedo_arr[:y_split, :] = base_skin

        # Clothing region with realistic shading gradient & ambient occlusion
        y_g, x_g = np.ogrid[:height, :width]
        grad = np.clip((y_g - y_split) / float(height - y_split), 0.0, 1.0)[:, :, np.newaxis]
        cloth_shaded = base_cloth * (1.05 - 0.20 * grad)
        albedo_arr[y_split:, :] = cloth_shaded[y_split:, :]

        # Trim & belt band
        belt_y0, belt_y1 = int(height * 0.52), int(height * 0.56)
        albedo_arr[belt_y0:belt_y1, :] = base_trim

        # Eye sockets, sclera, iris & pupils on Face UV Island
        eye_y = int(height * 0.457)
        for eye_x in [int(width * 0.35), int(width * 0.65)]:
            d_eye = np.sqrt((x_g - eye_x)**2 + (y_g - eye_y)**2)
            # Sclera / eye white
            sclera_mask = (d_eye < (width * 0.055))[:, :, np.newaxis]
            albedo_arr = np.where(sclera_mask, np.array([248, 248, 252], dtype=np.float32), albedo_arr)
            # Iris
            iris_mask = (d_eye < (width * 0.028))[:, :, np.newaxis]
            albedo_arr = np.where(iris_mask, np.array(eye_rgb, dtype=np.float32), albedo_arr)
            # Pupil
            pupil_mask = (d_eye < (width * 0.012))[:, :, np.newaxis]
            albedo_arr = np.where(pupil_mask, np.array([12, 12, 16], dtype=np.float32), albedo_arr)

        # Eyebrows
        eb_y = int(height * 0.375)
        for eb_x in [int(width * 0.35), int(width * 0.65)]:
            d_eb = np.sqrt(((x_g - eb_x) / 1.8)**2 + (y_g - eb_y)**2)
            eb_mask = (d_eb < (width * 0.038))[:, :, np.newaxis]
            albedo_arr = np.where(eb_mask, np.array(hair_rgb, dtype=np.float32), albedo_arr)

        # Lips with crisp vermilion contours
        lip_y = int(height * 0.772)
        lip_x = int(width * 0.50)
        d_lip = np.sqrt(((x_g - lip_x) / 2.0)**2 + (y_g - lip_y)**2)
        lip_mask = (d_lip < (width * 0.045))[:, :, np.newaxis]
        lip_color = np.array([160, 45, 60], dtype=np.float32) if is_ghost else np.array([195, 90, 95], dtype=np.float32)
        albedo_arr = np.where(lip_mask, lip_color, albedo_arr)

        # Localized Chest Emblem / Insignia (No hazard zebra stripes)
        if glow_rgb is not None:
            g_color = np.array(glow_rgb, dtype=np.float32)
            cx, cy = int(width * 0.50), int(y_split + (belt_y0 - y_split) * 0.35)
            # Centered diamond / heraldic crest
            d_crest = (np.abs(x_g - cx) * 1.2 + np.abs(y_g - cy))
            crest_mask = (d_crest < (width * 0.055)) & (d_crest > (width * 0.038))
            albedo_arr[crest_mask] = g_color

        # Horror Blood Spatter for ghost archetype
        if is_ghost:
            rng = np.random.RandomState(999)
            blood_color = np.array([125, 12, 20], dtype=np.float32)
            for _ in range(25):
                bx = rng.randint(int(width * 0.25), int(width * 0.75))
                by = rng.randint(int(height * 0.28), int(height * 0.85))
                br = rng.randint(10, 45)
                dist = np.sqrt((x_g - bx)**2 + (y_g - by)**2)
                splat = np.exp(-dist**2 / (2 * br**2))[:, :, np.newaxis]
                albedo_arr = albedo_arr * (1.0 - splat * 0.85) + blood_color * (splat * 0.85)

        # Bottom Hem Dirt Gradient
        dirt_color = np.array([45, 38, 30], dtype=np.float32)
        dirt_gradient = np.clip((np.arange(height) - (height * 0.70)) / (height * 0.30), 0.0, 1.0)
        dirt_mask = dirt_gradient[:, np.newaxis, np.newaxis]
        albedo_arr = albedo_arr * (1.0 - dirt_mask * 0.60) + dirt_color * (dirt_mask * 0.60)

        albedo_img = Image.fromarray(np.clip(albedo_arr, 0, 255).astype(np.uint8), mode="RGB")

        # 2. Tangent-Space Normal Map (Clean, tailored micro-surface, no zebra stripes)
        cloth_mask = (y_g >= y_split).astype(np.float32)
        rng_n = np.random.RandomState(123)
        micro_grain = rng_n.normal(0.0, 0.04, (height, width)).astype(np.float32)

        if mat_metal > 0.5:
            # Polished plate metal with subtle brushed texture
            h_map = micro_grain * 0.6
            # Clean plate border seams
            seam_y_1d = (np.abs(np.arange(height) - belt_y0) < 4) | (np.abs(np.arange(height) - y_split) < 4)
            h_map[seam_y_1d, :] -= 0.5
        else:
            # Natural soft vertical cloth folds
            h_map = micro_grain * 0.4 + (np.sin(x_g * 0.04) * 0.6 + np.sin(x_g * 0.09) * 0.3) * cloth_mask

        normal_img = self.height_to_normal_map(h_map, strength=3.5)

        # 3. Roughness Map
        rough_byte = int(np.clip(mat_rough * 255.0, 30, 240))
        roughness_arr = np.full((height, width), rough_byte, dtype=np.uint8)
        roughness_arr[:y_split, :] = 135  # Skin

        # Eyes glossy
        for eye_x in [int(width * 0.38), int(width * 0.62)]:
            d_eye = np.sqrt((x_g - eye_x)**2 + (y_g - eye_y)**2)
            roughness_arr[d_eye < (width * 0.035)] = 25  # Highly glossy pupil/iris

        roughness_img = Image.fromarray(roughness_arr, mode="L")

        # 4. Alpha Textures
        alpha_hair = self.generate_hair_alpha_texture(width=width, height=height, hair_rgb=hair_rgb)
        alpha_cloth = self.generate_torn_cloth_alpha(width=width, height=height, cloth_rgb=cloth_rgb, is_tattered=is_tattered)

        return PBRTextureSet(
            albedo=albedo_img,
            normal=normal_img,
            roughness=roughness_img,
            alpha_hair=alpha_hair,
            alpha_cloth=alpha_cloth
        )

        return PBRTextureSet(
            albedo=albedo_img,
            normal=normal_img,
            roughness=roughness_img,
            alpha_hair=alpha_hair,
            alpha_cloth=alpha_cloth
        )
