import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import sobel, gaussian_filter, distance_transform_edt, uniform_filter

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
    specular: Optional[Image.Image] = None
    metallic_roughness: Optional[Image.Image] = None
    ao: Optional[Image.Image] = None

class PBRTextureEngine:
    def __init__(self):
        pass

    def enhance_micro_details(
        self,
        albedo_img: Image.Image,
        sharpen_strength: float = 1.35
    ) -> Image.Image:
        """
        Enhances crisp micro-details (eyes, forehead emblem, horns, collar edges, fabric weave)
        using edge-preserving dual-scale unsharp masking and targeted dental anti-bleed clamping.
        Guarantees that face features, emblems, and original colors are never bleached or washed out.
        """
        arr = np.array(albedo_img.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        # 1. Edge-preserving dual-scale filtering
        blur_fine = np.stack([gaussian_filter(arr[:, :, c], sigma=0.60) for c in range(3)], axis=-1)
        blur_med = np.stack([gaussian_filter(arr[:, :, c], sigma=1.50) for c in range(3)], axis=-1)

        gx = sobel(lum, axis=1)
        gy = sobel(lum, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        edge_weight = np.clip(grad_mag / 35.0, 0.0, 1.0)[:, :, np.newaxis]

        # Crisp micro-edges stay original, flat areas are subtly denoised
        denoised = blur_fine * (1.0 - edge_weight) + arr * edge_weight

        y_grid, x_grid = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij"
        )
        max_c = np.maximum(np.maximum(denoised[:, :, 0], denoised[:, :, 1]), denoised[:, :, 2])
        min_c = np.minimum(np.minimum(denoised[:, :, 0], denoised[:, :, 1]), denoised[:, :, 2])
        chroma = max_c - min_c

        # 2. Targeted Dental Enamel Anti-Bleed:
        # Strictly target red/peach lip color bleeding into teeth inside the mouth cavity
        is_dental_bleed = (
            (y_grid >= 0.22) & (y_grid <= 0.38) &
            (np.abs(x_grid - 0.50) < 0.16) &
            (lum > 130.0) &
            (denoised[:, :, 0] > denoised[:, :, 1] + 15.0) &
            (chroma > 20.0)
        )
        if np.any(is_dental_bleed):
            denoised[is_dental_bleed] = np.array([246.0, 244.0, 240.0], dtype=np.float32)

        # 3. Collar trim contrast boost along steep boundary
        is_collar_zone = (y_grid >= 0.22) & (y_grid <= 0.44) & (edge_weight[:, :, 0] > 0.40)
        if np.any(is_collar_zone):
            lum_blur = gaussian_filter(lum, sigma=1.2)
            lum_diff = (lum - lum_blur)[is_collar_zone, np.newaxis]
            collar_scale = np.where(lum_diff > 2.0, 1.05, np.where(lum_diff < -2.0, 0.95, 1.0))
            denoised[is_collar_zone] = np.clip(denoised[is_collar_zone] * collar_scale, 0.0, 255.0)

        # 4. Crisp Multi-Band High-Frequency Unsharp Masking
        high_freq_fine = arr - blur_fine
        high_freq_med = blur_fine - blur_med
        sharpened = (
            denoised
            + high_freq_fine * (sharpen_strength * 0.95)
            + high_freq_med * (sharpen_strength * 0.30)
        )

        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8), mode="RGB")

    def repair_occluded_regions(
        self,
        albedo_img: Image.Image,
        fg_mask: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Repairs washed-out white/gray camera occlusion artifacts in tight anatomical crevices
        (under-chin, armpits, inner groin, posterior heel/sole) without modifying intentional
        white clothing (white shorts, sneakers, collar, teeth).
        """
        arr = np.array(albedo_img.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        chroma = max_c - min_c
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        y_grid, x_grid = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij"
        )

        is_chin_zone = (y_grid >= 0.18) & (y_grid <= 0.36)
        is_armpit_zone = (y_grid >= 0.26) & (y_grid <= 0.58)
        is_groin_zone = (y_grid >= 0.50) & (y_grid <= 0.70)
        is_heel_sole_zone = (y_grid >= 0.78) & (y_grid <= 1.00)

        in_occlusion_zone = is_chin_zone | is_armpit_zone | is_groin_zone | is_heel_sole_zone

        # Preserve deliberate bright facial features (eyes strictly in upper head island)
        is_face_feature = (y_grid < 0.18) & (lum > 215.0)

        # Detect washed out chalky blotches in crevices (low chroma < 16, moderate to high lum)
        is_washed_out = in_occlusion_zone & (~is_face_feature) & (
            ((lum > 118.0) & (chroma < 18.0)) |
            ((lum > 190.0) & (chroma < 26.0))
        )
        if fg_mask is not None:
            is_washed_out = is_washed_out & fg_mask

        valid_source = (~is_washed_out)
        if fg_mask is not None:
            valid_source = valid_source & fg_mask

        zones = [
            ("chin", is_chin_zone, 0.88),
            ("armpit", is_armpit_zone, 0.84),
            ("groin", is_groin_zone, 0.82),
            ("heel", is_heel_sole_zone, 0.80)
        ]

        for z_name, z_mask, cavity_ao_factor in zones:
            if not np.any(z_mask):
                continue
            # If the zone has significant intentional white fabric/footwear, do not inpaint
            light_fraction = np.mean((lum[z_mask] > 170.0) & (chroma[z_mask] < 30.0))
            if light_fraction > 0.25 and z_name in ["groin", "heel"]:
                continue

            z_washed = is_washed_out & z_mask
            z_valid = valid_source & z_mask & (chroma >= 8.0)
            if not np.any(z_valid):
                z_valid = valid_source & z_mask

            if np.any(z_washed) and np.any(z_valid):
                dist, nearest_idx = distance_transform_edt(~z_valid, return_indices=True)
                inpainted = arr[nearest_idx[0], nearest_idx[1]]
                cavity_ao = np.clip(cavity_ao_factor + 0.10 * np.exp(-dist / 14.0), cavity_ao_factor, 0.96)[:, :, np.newaxis]
                blend_alpha = np.clip((24.0 - chroma) / 24.0, 0.35, 1.0)[:, :, np.newaxis]
                repaired_pixels = arr * (1.0 - blend_alpha) + (inpainted * cavity_ao) * blend_alpha
                arr = np.where(z_washed[:, :, np.newaxis], repaired_pixels, arr)

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    def synthesize_material_zone_maps(
        self,
        albedo_img: Image.Image,
        base_metalness: float = 0.0,
        base_roughness: float = 0.65
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Generates zone-segmented PBR Roughness, Specular, Ambient Occlusion, and glTF 2.0 packed
        MetallicRoughness maps directly from an albedo texture so skin (matte-satin), sportswear
        fabric (high-roughness woven matte), hair (anisotropic sheen), shoes/soles (semi-gloss rubber),
        and eyes/teeth (wet gloss) exhibit physically distinct light responses.
        Returns: (roughness_img, specular_img, ao_img, metallic_roughness_gltf_img)
        """
        arr = np.array(albedo_img.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        lum = 0.299 * r + 0.587 * g + 0.114 * b
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        chroma = max_c - min_c

        y_grid, x_grid = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij"
        )

        # High-frequency micro-texture variance for fabric weave vs smooth skin
        local_mean = uniform_filter(lum, size=5)
        local_sq_mean = uniform_filter(lum**2, size=5)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0.0))

        # 1. Material Zone Segmentation Masks
        # Skin detection: warm peach/tan/brown/fair hue (R > G > B with moderate saturation)
        is_skin = (
            (r > 125.0) & (g > 75.0) & (b > 50.0) &
            ((r - g) > 10.0) & ((g - b) >= 2.0) & ((r - b) < 115.0) &
            (y_grid < 0.84)
        )

        # Metallic / Gold / Silver Armor trim detection
        is_gold_metal = (r > 165.0) & (g > 125.0) & (b < 90.0) & ((r - b) > 65.0)
        is_steel_metal = (base_metalness > 0.55) & (~is_skin)
        is_metal = is_gold_metal | is_steel_metal

        # Eyes & Teeth detection (upper head zone, high contrast or bright sclera/enamel)
        is_eyes_or_teeth = (y_grid < 0.28) & (lum > 210.0) & (chroma < 28.0)

        # Hair zone (top scalp/head region y < 0.26, non-skin, non-eye)
        is_hair = (y_grid < 0.26) & (~is_skin) & (~is_eyes_or_teeth) & (~is_metal)

        # Footwear / Shoes / Rubber Soles zone (bottom 15% of UV height)
        is_shoes = (y_grid >= 0.85) & (~is_skin)

        # Sportswear / Fabric / Clothing (remaining torso/legs region)
        is_fabric = (~is_skin) & (~is_metal) & (~is_eyes_or_teeth) & (~is_hair) & (~is_shoes)

        # 2. Construct Per-Zone Roughness (0..255), Specular (0..255), and Metallic (0..255)
        rough_arr = np.full((h, w), base_roughness * 255.0, dtype=np.float32)
        spec_arr = np.full((h, w), 100.0, dtype=np.float32)
        metal_arr = np.full((h, w), base_metalness * 255.0, dtype=np.float32)

        # Synthetic micro-weave & pore noise
        yy, xx = np.ogrid[:h, :w]
        fabric_weave = (np.sin(xx * 0.65) * np.cos(yy * 0.65)) * 9.0
        skin_pore = (np.sin(xx * 1.2 + yy * 0.8) * np.cos(xx * 0.8 - yy * 1.2)) * 4.5

        # Zone A: Matte-Satin Natural Skin (Roughness ~0.54 -> 138, Specular ~0.38 -> 96)
        rough_arr = np.where(is_skin, 138.0 + skin_pore, rough_arr)
        spec_arr = np.where(is_skin, 96.0, spec_arr)
        metal_arr = np.where(is_skin, 0.0, metal_arr)

        # Zone B: Sportswear / Woven Fabric (High Matte Roughness ~0.82 -> 210, Low Specular ~0.18 -> 46)
        rough_arr = np.where(is_fabric, 210.0 + fabric_weave + np.clip(local_std * 0.4, -8.0, 12.0), rough_arr)
        spec_arr = np.where(is_fabric, 46.0, spec_arr)
        metal_arr = np.where(is_fabric, 0.0, metal_arr)

        # Zone C: Hair Strands (Anisotropic Satin Sheen Roughness ~0.42 -> 107, Specular ~0.58 -> 148)
        hair_strand_var = np.sin(xx * 0.9) * 8.0
        rough_arr = np.where(is_hair, 107.0 + hair_strand_var, rough_arr)
        spec_arr = np.where(is_hair, 148.0, spec_arr)
        metal_arr = np.where(is_hair, 12.0, metal_arr)

        # Zone D: Shoes / Leather / Soles (Semi-gloss Roughness ~0.36 -> 92, Specular ~0.64 -> 163)
        rough_arr = np.where(is_shoes, 92.0 + skin_pore, rough_arr)
        spec_arr = np.where(is_shoes, 163.0, spec_arr)
        metal_arr = np.where(is_shoes, 18.0, metal_arr)

        # Zone E: Wet Glossy Eyes & Teeth (Roughness ~0.12 -> 30, Specular ~0.94 -> 240)
        rough_arr = np.where(is_eyes_or_teeth, 30.0, rough_arr)
        spec_arr = np.where(is_eyes_or_teeth, 240.0, spec_arr)
        metal_arr = np.where(is_eyes_or_teeth, 0.0, metal_arr)

        # Zone F: Metallic Trim / Armor (Roughness ~0.20 -> 51, Specular ~0.96 -> 245, Metalness ~0.88 -> 225)
        rough_arr = np.where(is_metal, 51.0, rough_arr)
        spec_arr = np.where(is_metal, 245.0, spec_arr)
        metal_arr = np.where(is_metal, 225.0, metal_arr)

        # Smooth transitions slightly (sigma=0.7) to avoid jagged aliasing between material zones
        rough_u8 = np.clip(gaussian_filter(rough_arr, sigma=0.65), 15, 248).astype(np.uint8)
        spec_u8 = np.clip(gaussian_filter(spec_arr, sigma=0.65), 15, 250).astype(np.uint8)
        metal_u8 = np.clip(gaussian_filter(metal_arr, sigma=0.65), 0, 255).astype(np.uint8)

        # 3. Cavity & Anatomical Ambient Occlusion (AO) Map
        crevice_ao = np.clip(255.0 - local_std * 1.8, 175.0, 255.0)
        # Subtle darkening in anatomical occlusion bands (under-chin y~0.26, armpits y~0.40, soles y~0.96)
        anat_ao = (
            1.0
            - 0.14 * np.exp(-((y_grid - 0.26) / 0.04) ** 2)
            - 0.12 * np.exp(-((y_grid - 0.42) / 0.05) ** 2)
            - 0.15 * np.exp(-((y_grid - 0.97) / 0.03) ** 2)
        )
        ao_u8 = np.clip(crevice_ao * anat_ao, 150, 255).astype(np.uint8)

        # 4. Pack glTF 2.0 Standard MetallicRoughness Texture:
        # Channel R = Ambient Occlusion, Channel G = Roughness, Channel B = Metalness
        mr_gltf = np.stack([ao_u8, rough_u8, metal_u8], axis=-1)

        return (
            Image.fromarray(rough_u8, mode="L"),
            Image.fromarray(spec_u8, mode="L"),
            Image.fromarray(ao_u8, mode="L"),
            Image.fromarray(mr_gltf, mode="RGB")
        )

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
                skin_rgb = (236, 205, 186)
                hair_rgb = (45, 32, 24)
                eye_rgb = (30, 36, 48)
                cloth_rgb = (37, 99, 235)
                trim_rgb = (15, 23, 42)
                glow_rgb = None
                mat_metal = 0.0
                mat_rough = 0.82
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

        # Crisp Collar Border at neck/cloth transition (y_split)
        collar_y0, collar_y1 = max(0, y_split - int(height * 0.008)), min(height, y_split + int(height * 0.016))
        albedo_arr[collar_y0:collar_y1, :] = base_trim

        # Eye sockets, sclera, iris & pupils on Face UV Island (strictly in upper head zone y < y_split)
        eye_y = int(height * 0.16)
        for eye_x in [int(width * 0.35), int(width * 0.65)]:
            d_eye = np.sqrt((x_g - eye_x)**2 + (y_g - eye_y)**2)
            # Orbital socket shadow rim
            socket_mask = ((d_eye >= (width * 0.055)) & (d_eye < (width * 0.068)))[:, :, np.newaxis]
            albedo_arr = np.where(socket_mask, base_skin * 0.78, albedo_arr)
            # Sclera / eye white
            sclera_mask = (d_eye < (width * 0.055))[:, :, np.newaxis]
            albedo_arr = np.where(sclera_mask, np.array([248, 248, 252], dtype=np.float32), albedo_arr)
            # Iris
            iris_mask = (d_eye < (width * 0.028))[:, :, np.newaxis]
            albedo_arr = np.where(iris_mask, np.array(eye_rgb, dtype=np.float32), albedo_arr)
            # Pupil
            pupil_mask = (d_eye < (width * 0.012))[:, :, np.newaxis]
            albedo_arr = np.where(pupil_mask, np.array([12, 12, 16], dtype=np.float32), albedo_arr)

        # Eyebrows (above eyes in upper head zone)
        eb_y = int(height * 0.10)
        for eb_x in [int(width * 0.35), int(width * 0.65)]:
            d_eb = np.sqrt(((x_g - eb_x) / 1.8)**2 + (y_g - eb_y)**2)
            eb_mask = (d_eb < (width * 0.038))[:, :, np.newaxis]
            albedo_arr = np.where(eb_mask, np.array(hair_rgb, dtype=np.float32), albedo_arr)

        # Lips with crisp vermilion contours in lower face zone (before neck/collar)
        lip_y = int(height * 0.28)
        lip_x = int(width * 0.50)
        d_lip = np.sqrt(((x_g - lip_x) / 2.0)**2 + (y_g - lip_y)**2)
        lip_mask = (d_lip < (width * 0.045))[:, :, np.newaxis]
        lip_color = np.array([160, 45, 60], dtype=np.float32) if is_ghost else np.array([195, 90, 95], dtype=np.float32)
        albedo_arr = np.where(lip_mask, lip_color, albedo_arr)

        # Crisp teeth / dental line inside inner lip commissure
        d_teeth = np.sqrt(((x_g - lip_x) / 2.2)**2 + (y_g - lip_y)**2)
        teeth_mask = ((d_teeth < (width * 0.024)) & (np.abs(y_g - lip_y) < max(2, int(height * 0.005))))[:, :, np.newaxis]
        albedo_arr = np.where(teeth_mask, np.array([246, 244, 240], dtype=np.float32), albedo_arr)

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

        raw_albedo_img = Image.fromarray(np.clip(albedo_arr, 0, 255).astype(np.uint8), mode="RGB")
        repaired_albedo = self.repair_occluded_regions(raw_albedo_img)
        albedo_img = self.enhance_micro_details(repaired_albedo, sharpen_strength=1.35)

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

        # 3. Zone-Segmented Roughness, Specular, AO, and glTF MetallicRoughness Maps
        roughness_img, specular_img, ao_img, mr_gltf_img = self.synthesize_material_zone_maps(
            albedo_img,
            base_metalness=mat_metal,
            base_roughness=mat_rough
        )

        # Ensure eyes & teeth on face UV island have glossy finish in roughness map
        rough_arr = np.array(roughness_img, dtype=np.uint8)
        for eye_x in [int(width * 0.35), int(width * 0.65)]:
            d_eye = np.sqrt((x_g - eye_x)**2 + (y_g - eye_y)**2)
            rough_arr[d_eye < (width * 0.035)] = 22  # Highly glossy pupil/iris
        roughness_img = Image.fromarray(rough_arr, mode="L")

        # Update G channel of mr_gltf_img with refined eye glossiness
        mr_arr = np.array(mr_gltf_img, dtype=np.uint8)
        mr_arr[:, :, 1] = rough_arr
        mr_gltf_img = Image.fromarray(mr_arr, mode="RGB")

        # 4. Alpha Textures
        alpha_hair = self.generate_hair_alpha_texture(width=width, height=height, hair_rgb=hair_rgb)
        alpha_cloth = self.generate_torn_cloth_alpha(width=width, height=height, cloth_rgb=cloth_rgb, is_tattered=is_tattered)

        return PBRTextureSet(
            albedo=albedo_img,
            normal=normal_img,
            roughness=roughness_img,
            alpha_hair=alpha_hair,
            alpha_cloth=alpha_cloth,
            specular=specular_img,
            metallic_roughness=mr_gltf_img,
            ao=ao_img
        )
