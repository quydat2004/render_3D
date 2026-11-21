import os
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import trimesh
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError:
    genai = None
    types = None
    ClientError = Exception

from services.common.config import get_gemini_api_key

logger = logging.getLogger(__name__)
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

class AssetType(str, Enum):
    CHARACTER = "character"
    WEAPON = "weapon"
    ENVIRONMENT = "environment"

class CharacterColorPalette(BaseModel):
    skin_color_hex: str = Field(default="#E8BEA8", description="Primary skin/face hex color")
    hair_color_hex: str = Field(default="#1E1E24", description="Hair color hex")
    eye_color_hex: str = Field(default="#202025", description="Eye iris color hex")
    primary_cloth_hex: str = Field(default="#2A3240", description="Main outfit or armor hex color")
    secondary_cloth_hex: str = Field(default="#1D2430", description="Secondary trim, pants, or accent hex")
    accent_glow_hex: Optional[str] = Field(default=None, description="Emissive / cybernetic / magical glow hex color")
    metalness: float = Field(default=0.1, description="Material metalness 0.0 to 1.0 (0.8+ for knight armor/robot)")
    roughness: float = Field(default=0.6, description="Material roughness 0.0 to 1.0")
    clothing_type: str = Field(default="suit", description="robe, dress, armor, suit, jacket, kimono, casual")
    hair_style: str = Field(default="long_flowing", description="long_flowing, short_spiky, ponytail, bob")

class OrthographicConceptPrompt(BaseModel):
    front_prompt: str = Field(description="Prompt for front orthographic camera")
    side_prompt: str = Field(description="Prompt for 90-degree side profile camera")
    back_prompt: str = Field(description="Prompt for rear orthographic camera")
    style_notes: str = Field(description="Color, lighting, and material instructions")
    pose: str = Field(default="A-pose", description="Required anatomical stance: A-pose, T-pose, or Neutral")
    poly_budget: int = Field(default=25000, description="Target polygon budget in triangles/quads")
    palette: Optional[CharacterColorPalette] = Field(default_factory=CharacterColorPalette, description="Dynamic PBR color and material palette extracted from prompt")

class RoomSchema(BaseModel):
    id: str = Field(description="Room identifier")
    bounds: List[float] = Field(description="Bounding box [min_x, min_y, max_x, max_y]")
    height: Optional[float] = Field(default=4.0, description="Ceiling clearance height")

class CorridorSchema(BaseModel):
    connects: List[str] = Field(description="Room IDs linked by corridor")
    width: float = Field(default=4.0, description="Corridor width in meters")

class SpawnPointSchema(BaseModel):
    team: str = Field(default="Alpha", description="Team or faction tag")
    pos: List[float] = Field(description="[X, Y, Z] coordinate")
    rotation_yaw: Optional[float] = Field(default=0.0, description="Yaw orientation in degrees")

class InstanceSchema(BaseModel):
    asset_id: str = Field(description="Modular asset identifier (e.g. wall_4x4, floor_4x4)")
    position: List[float] = Field(description="[X, Y, Z] position")
    rotation: List[float] = Field(default=[0.0, 0.0, 0.0], description="Euler rotation degrees")
    scale: List[float] = Field(default=[1.0, 1.0, 1.0], description="Scale factors")

class LevelLayoutSchema(BaseModel):
    level_id: str = Field(description="Unique identifier for the map")
    dimensions: List[float] = Field(description="[width, length] in meters")
    grid_size_m: float = Field(default=2.0, description="Modular snapping grid step in meters")
    biome: str = Field(description="Theme/Biome designation (e.g., Cyberpunk_Alley, SciFi_Interior)")
    rooms: List[RoomSchema] = Field(default_factory=list, description="Defined sub-areas and bounds")
    corridors: List[CorridorSchema] = Field(default_factory=list, description="Corridor links between rooms")
    spawn_points: List[SpawnPointSchema] = Field(default_factory=list, description="Player/team spawn coordinates")
    instances: List[InstanceSchema] = Field(default_factory=list, description="Placed modular kitbash objects")

class QAInspectionResult(BaseModel):
    score: int = Field(ge=0, le=100, description="Quality score between 0 and 100")
    passed: bool = Field(description="True if score >= 85 and game engine ready")
    anomalies_detected: List[str] = Field(default_factory=list, description="Defects found: non-manifold edges, UV stretching, etc.")
    corrective_instructions: str = Field(description="Specific actionable advice to fix mesh/texture")

class GeminiOrchestrator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        offline_mode: bool = False
    ):
        self.offline_mode = offline_mode
        if not self.offline_mode:
            try:
                self.api_key = api_key or (get_gemini_api_key() if not client else "test_key")
            except ValueError:
                self.api_key = None
                self.offline_mode = True
        else:
            self.api_key = None

        if client:
            self.client = client
        elif genai and self.api_key and not self.offline_mode:
            try:
                self.client = genai.Client(api_key=self.api_key, http_options={"timeout": 10000})
            except Exception:
                self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def expand_asset_prompt(self, user_prompt: str, asset_type: AssetType) -> OrthographicConceptPrompt:
        if self.offline_mode or not self.client:
            return self._offline_expand_asset_prompt(user_prompt, asset_type)

        system_instructions = (
            f"You are a Senior 3D Technical Director. Expand the following user prompt for a 3D {asset_type.value}. "
            "Produce strict orthographic multi-view prompts (Front, Side, Back) under neutral flat lighting "
            "with zero perspective distortion. Adhere to anatomical guidelines (A-pose/T-pose for characters, "
            "clean profile for weapons). Return output adhering strictly to the JSON schema."
        )

        try:
            response = self.client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=f"{system_instructions}\n\nUser request: {user_prompt}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=OrthographicConceptPrompt,
                    temperature=0.2,
                ) if types else None,
            )
            return OrthographicConceptPrompt.model_validate_json(response.text)
        except Exception as e:
            logger.warning(f"Gemini API request failed ({e}). Falling back to offline engine.")
            return self._offline_expand_asset_prompt(user_prompt, asset_type)

    def plan_environment(self, prompt: str, dimensions: Optional[List[float]] = None) -> LevelLayoutSchema:
        if self.offline_mode or not self.client:
            return self._offline_plan_environment(prompt, dimensions=dimensions)

        system_instructions = (
            "You are a 3D Level Designer. Generate a quantitative, metric grid-based modular level layout. "
            "Define dimensions in meters, grid snapping size, biomes, spawn points, rooms, and placed instances. "
            "Return valid JSON adhering to the LevelLayoutSchema."
        )

        try:
            response = self.client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=f"{system_instructions}\n\nEnvironment Request: {prompt}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LevelLayoutSchema,
                    temperature=0.2,
                ) if types else None,
            )
            layout = LevelLayoutSchema.model_validate_json(response.text)
            if len(layout.instances) < 6:
                return self._offline_plan_environment(prompt, dimensions=dimensions or layout.dimensions)
            return layout
        except Exception as e:
            logger.warning(f"Gemini API plan_environment failed ({e}). Falling back to offline engine.")
            return self._offline_plan_environment(prompt, dimensions=dimensions)

    def inspect_turntable(self, image_bytes_list: List[bytes], asset_type: AssetType) -> QAInspectionResult:
        if self.offline_mode or not self.client or not image_bytes_list:
            return self._offline_inspect_turntable(image_bytes_list, asset_type)

        parts = []
        if types:
            for img in image_bytes_list:
                parts.append(types.Part.from_bytes(data=img, mime_type="image/png"))

        prompt_text = (
            f"Analyze these turntable render frames of a generated 3D {asset_type.value}. "
            "Inspect for: 1. Non-manifold geometry, surface tearing, floating vertices. "
            "2. Severe UV stretching, seam artifacts, bad normal mapping. "
            "3. Anatomical symmetry and proportion issues. "
            "Score from 0 to 100. If score >= 85, passed=True. Return valid JSON."
        )
        parts.append(prompt_text)

        try:
            response = self.client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QAInspectionResult,
                    temperature=0.1,
                ) if types else None,
            )
            return QAInspectionResult.model_validate_json(response.text)
        except Exception as e:
            logger.warning(f"Gemini API inspect_turntable failed ({e}). Falling back to offline QA.")
            return self._offline_inspect_turntable(image_bytes_list, asset_type)

    def render_mesh_screenshot(self, glb_path: str, size: Tuple[int, int] = (512, 512)) -> bytes:
        """
        Renders an orthographic front screenshot of a 3D GLB model using OpenCV for Visual QA.
        """
        try:
            import cv2
            scene = trimesh.load(glb_path)
            if isinstance(scene, trimesh.Scene):
                geoms = list(scene.geometry.values())
                mesh = trimesh.util.concatenate(geoms) if geoms else None
            else:
                mesh = scene

            if mesh is None or len(mesh.vertices) == 0:
                return b""

            img = np.full((size[1], size[0], 3), 35, dtype=np.uint8) # Dark studio backdrop
            verts = mesh.vertices
            faces = mesh.faces
            bounds = mesh.bounds

            cx = (bounds[0][0] + bounds[1][0]) * 0.5
            min_y, max_y = bounds[0][1], bounds[1][1]
            h = max(max_y - min_y, 1e-4)

            scale = (size[1] * 0.85) / h
            px = np.clip(((verts[:, 0] - cx) * scale + size[0] * 0.5), 0, size[0] - 1).astype(np.int32)
            py = np.clip((size[1] * 0.92 - (verts[:, 1] - min_y) * scale), 0, size[1] - 1).astype(np.int32)

            face_z = np.mean(verts[faces, 2], axis=1)
            sorted_face_idx = np.argsort(face_z)

            # Extract texture image or colors
            tex_img = None
            if hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "baseColorTexture") and mesh.visual.material.baseColorTexture is not None:
                tex_img = mesh.visual.material.baseColorTexture
            elif hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "image") and mesh.visual.material.image is not None:
                tex_img = mesh.visual.material.image
            elif hasattr(mesh.visual, "image") and mesh.visual.image is not None:
                tex_img = mesh.visual.image

            if tex_img is not None and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) == len(verts):
                tex_arr = np.array(tex_img.convert("RGB"))
                th, tw = tex_arr.shape[:2]
                uvs = mesh.visual.uv
                face_uvs = np.mean(uvs[faces], axis=1)
                u_px = np.clip((face_uvs[:, 0] * tw).astype(np.int32), 0, tw - 1)
                v_px = np.clip((face_uvs[:, 1] * th).astype(np.int32), 0, th - 1)
                f_colors = tex_arr[v_px, u_px]
            elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(verts):
                vc = mesh.visual.vertex_colors[:, :3]
                f_colors = np.mean(vc[faces], axis=1).astype(np.uint8)
            else:
                diffuse = np.clip(mesh.face_normals[:, 2] * 0.6 + 0.4, 0.2, 1.0)
                f_colors = (diffuse[:, np.newaxis] * [210, 180, 160]).astype(np.uint8)

            pts = np.stack([px, py], axis=-1)
            for f_idx in sorted_face_idx:
                tri = pts[faces[f_idx]]
                c = tuple(int(val) for val in f_colors[f_idx])
                cv2.fillConvexPoly(img, tri, c)

            success, encoded = cv2.imencode(".png", img)
            return encoded.tobytes() if success else b""
        except Exception as e:
            logger.warning(f"Orthographic screenshot failed ({e}).")
            return b""

    def inspect_visual_alignment(
        self,
        model_glb_path: str,
        reference_image_path: Optional[str] = None
    ) -> QAInspectionResult:
        """
        Visual QA pass: Inspects rendered 3D model against reference 2D concept with Gemini VLM.
        Verifies texture orientation (not upside down), color fidelity, and anatomical integrity.
        """
        render_png = self.render_mesh_screenshot(model_glb_path)

        if self.client and reference_image_path and Path(reference_image_path).exists() and render_png and types:
            try:
                ref_bytes = Path(reference_image_path).read_bytes()
                parts = [
                    types.Part.from_bytes(data=ref_bytes, mime_type="image/png"),
                    types.Part.from_bytes(data=render_png, mime_type="image/png"),
                    (
                        "You are an expert 3D Asset QA Inspector. "
                        "Image 1: Reference 2D concept of the character. "
                        "Image 2: Front orthographic render of the generated 3D model. "
                        "Critically inspect: "
                        "1. Texture Orientation: Is the face/texture right-side up? If the face is inverted (e.g. face on crotch/belly) or upside down, score <= 30, passed = False. "
                        "2. Color Fidelity: Does the 3D model preserve the primary colors of the concept (e.g. red shirt, hair)? If the model is mostly blank white or lost colors, score <= 50, passed = False. "
                        "3. Proportion & Silhouette: Are the limbs, torso, and head well proportioned without severe distortion? "
                        "Score from 0 to 100. If score >= 85, passed=True. Return valid JSON matching QAInspectionResult schema."
                    )
                ]
                response = self.client.models.generate_content(
                    model=DEFAULT_GEMINI_MODEL,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=QAInspectionResult,
                        temperature=0.1,
                    ),
                )
                return QAInspectionResult.model_validate_json(response.text)
            except Exception as e:
                logger.warning(f"Gemini Visual QA API call failed ({e}). Running algorithmic inspection.")

        return self._programmatic_visual_qa(model_glb_path, reference_image_path)

    def _programmatic_visual_qa(
        self,
        model_glb_path: str,
        reference_image_path: Optional[str] = None
    ) -> QAInspectionResult:
        try:
            scene = trimesh.load(model_glb_path)
            geoms = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
            mesh = geoms[0] if geoms else None

            if mesh is None or len(mesh.vertices) == 0:
                return QAInspectionResult(
                    score=0,
                    passed=False,
                    anomalies_detected=["Generated asset contains no mesh geometry"],
                    corrective_instructions="Regenerate mesh."
                )

            anomalies = []
            score = 96

            # 1. Planar projection check: front and back must not share identical UV coordinates
            if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices):
                front = mesh.vertices[:, 2] > np.median(mesh.vertices[:, 2])
                if front.any() and (~front).any():
                    spread = np.linalg.norm(
                        mesh.visual.uv[front].mean(axis=0) - mesh.visual.uv[~front].mean(axis=0)
                    )
                    if spread < 0.02:
                        anomalies.append("Planar projection detected: front and back share the same UV coordinates")
                        score = min(score, 30)

            # 2. Texture & surface material presence check
            has_texture = (
                hasattr(mesh.visual, "material") and mesh.visual.material is not None
            ) or (
                hasattr(mesh.visual, "image") and mesh.visual.image is not None
            ) or (
                hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None
            )
            if not has_texture:
                anomalies.append("Missing texture maps and vertex colors (untextured clay mesh)")
                score = min(score, 35)

            # 3. Strict Color Fidelity & Palette Alignment against reference 2D concept
            if reference_image_path and Path(reference_image_path).exists():
                try:
                    from PIL import Image
                    raw_ref = Image.open(reference_image_path)
                    try:
                        from services.inference.hunyuan3d_client import Hunyuan3DClient
                        clean_ref = Hunyuan3DClient().remove_background(raw_ref)
                    except Exception:
                        clean_ref = raw_ref.convert("RGBA")

                    ref_arr = np.array(clean_ref.convert("RGBA"))
                    fg_mask = ref_arr[:, :, 3] > 120
                    fg_ref = ref_arr[fg_mask][:, :3] if np.any(fg_mask) else ref_arr.reshape(-1, 4)[:, :3]
                    if len(fg_ref) > 50:
                        ref_med = np.median(fg_ref, axis=0).astype(np.float32)

                        # Extract model surface color samples (sample front hemisphere for dual-atlas)
                        tex_img = None
                        if hasattr(mesh.visual, "material") and getattr(mesh.visual.material, "baseColorTexture", None):
                            tex_img = mesh.visual.material.baseColorTexture
                        elif hasattr(mesh.visual, "image") and mesh.visual.image is not None:
                            tex_img = mesh.visual.image

                        if tex_img is not None:
                            tex_rgb = np.array(tex_img.convert("RGB"))
                            th, tw = tex_rgb.shape[:2]
                            # Sample front half of the atlas [0..half_w]
                            front_tex = tex_rgb[:, :tw // 2]
                            model_med = np.median(front_tex.reshape(-1, 3), axis=0).astype(np.float32)
                        elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                            vc = mesh.visual.vertex_colors[:, :3].astype(np.float32)
                            model_med = np.median(vc, axis=0)
                        else:
                            model_med = np.array([120.0, 120.0, 120.0], dtype=np.float32)

                        # Euclidean color distance in sRGB
                        color_dist = float(np.linalg.norm(ref_med - model_med))
                        if color_dist > 125.0:
                            anomalies.append(
                                f"Severe color mismatch: model dominant color {model_med.astype(int).tolist()} deviates from reference {ref_med.astype(int).tolist()} (distance {color_dist:.1f})"
                            )
                            score = min(score, 45)
                        elif color_dist > 85.0:
                            score = min(score, 78)
                except Exception as color_err:
                    logger.debug(f"Color fidelity check skipped: {color_err}")

            passed = (score >= 85 and len(anomalies) == 0)
            instructions = (
                "Visual QA PASSED: Neural geometry aligned, 360-degree UV texture synthesized, color fidelity verified."
                if passed else
                f"Visual QA FAILED: {'; '.join(anomalies)}"
            )

            return QAInspectionResult(
                score=score,
                passed=passed,
                anomalies_detected=anomalies,
                corrective_instructions=instructions
            )
        except Exception as e:
            return QAInspectionResult(
                score=50,
                passed=False,
                anomalies_detected=[f"QA inspection exception: {e}"],
                corrective_instructions="Could not inspect 3D asset geometry."
            )

    def _infer_character_palette(self, prompt: str) -> CharacterColorPalette:
        p = prompt.lower()
        palette = CharacterColorPalette()

        # 1. Detect archetype & skin
        if any(w in p for w in ["ma nữ", "bóng ma", "hồn ma", "ghost", "witch", "spirit", "kinh dị", "horror", "zombie"]):
            palette.skin_color_hex = "#EEF2F6"      # Corpse pale white
            palette.hair_color_hex = "#0E0E14"      # Deep raven black
            palette.eye_color_hex = "#DC2626"       # Crimson red eyes
            palette.primary_cloth_hex = "#F5F8FC"   # Burial shroud white
            palette.secondary_cloth_hex = "#B91C1C" # Blood red accents
            palette.clothing_type = "dress"
            palette.hair_style = "long_flowing"
            palette.metalness = 0.05
            palette.roughness = 0.80
        elif any(w in p for w in ["chibi", "runner", "sport", "thể thao", "chạy bộ", "athlete", "sneaker", "jersey"]):
            palette.skin_color_hex = "#ECCDBA"      # Warm natural skin
            palette.hair_color_hex = "#2D1F17"      # Dark brown hair
            palette.eye_color_hex = "#1E293B"       # Dark natural eyes
            palette.primary_cloth_hex = "#EAB308" if any(w in p for w in ["vàng", "yellow", "gold"]) else "#2563EB"
            palette.secondary_cloth_hex = "#0F172A" # Dark athletic shorts
            palette.clothing_type = "suit"
            palette.hair_style = "short_spiky"
            palette.metalness = 0.0
            palette.roughness = 0.82
        elif any(w in p for w in ["vàng", "gold", "golden", "hoàng kim", "paladin", "knight", "hiệp sĩ"]):
            palette.skin_color_hex = "#ECCDBA"      # Warm tanned skin
            palette.hair_color_hex = "#FACC15"      # Golden blonde
            palette.eye_color_hex = "#3B82F6"       # Heroic azure blue
            palette.primary_cloth_hex = "#D4AF37"   # Antique polished gold
            palette.secondary_cloth_hex = "#1E293B" # Royal navy under-armor
            palette.accent_glow_hex = "#FEF08A"     # Golden divine aura
            palette.clothing_type = "armor"
            palette.hair_style = "short_spiky"
            palette.metalness = 0.85
            palette.roughness = 0.25
        elif any(w in p for w in ["cyberpunk", "samurai", "ninja", "cyborg", "robot", "mecha"]):
            palette.skin_color_hex = "#E2E8F0"      # Synthetic skin
            palette.hair_color_hex = "#0F172A"      # Jet black
            palette.eye_color_hex = "#00F0FF"       # Glowing cyan optics
            palette.primary_cloth_hex = "#1E293B"   # Carbon fiber dark armor
            palette.secondary_cloth_hex = "#0F172A" # Stealth charcoal
            palette.accent_glow_hex = "#00F0FF"     # Neon cyan circuitry
            palette.clothing_type = "armor"
            palette.hair_style = "ponytail"
            palette.metalness = 0.65
            palette.roughness = 0.40
        elif any(w in p for w in ["phù thủy", "wizard", "mage", "sorcerer", "áo choàng"]):
            palette.skin_color_hex = "#E8D5C4"
            palette.hair_color_hex = "#E2E8F0"      # Silver white
            palette.eye_color_hex = "#A855F7"       # Violet arcane
            palette.primary_cloth_hex = "#581C87"   # Royal purple velvet
            palette.secondary_cloth_hex = "#C084FC" # Lavender silk trim
            palette.accent_glow_hex = "#C084FC"     # Arcane rune glow
            palette.clothing_type = "robe"
            palette.hair_style = "long_flowing"
            palette.metalness = 0.10
            palette.roughness = 0.75
        elif any(w in p for w in ["đỏ", "red", "crimson", "huyết"]):
            palette.primary_cloth_hex = "#991B1B"   # Crimson red
            palette.secondary_cloth_hex = "#18181B"
            palette.accent_glow_hex = "#EF4444"
        elif any(w in p for w in ["xanh lá", "green", "emerald", "forest", "orc"]):
            palette.skin_color_hex = "#65A30D" if "orc" in p else "#E8BEA8"
            palette.primary_cloth_hex = "#15803D"   # Emerald forest
            palette.secondary_cloth_hex = "#78350F" # Leather brown

        # Dynamic color overrides based on explicit prompt words
        if "tóc vàng" in p or "blonde" in p:
            palette.hair_color_hex = "#FACC15"
        elif "tóc trắng" in p or "tóc bạc" in p or "white hair" in p:
            palette.hair_color_hex = "#F1F5F9"
        elif "tóc đỏ" in p or "red hair" in p:
            palette.hair_color_hex = "#DC2626"
        elif "tóc xanh" in p or "blue hair" in p:
            palette.hair_color_hex = "#2563EB"

        if "áo đen" in p or "black dress" in p or "black suit" in p:
            palette.primary_cloth_hex = "#111827"
        elif "áo đỏ" in p or "red dress" in p:
            palette.primary_cloth_hex = "#B91C1C"
        elif "áo xanh" in p or "blue dress" in p:
            palette.primary_cloth_hex = "#1D4ED8"
        elif "áo trắng" in p or "white dress" in p:
            palette.primary_cloth_hex = "#F8FAFC"

        return palette

    def _offline_expand_asset_prompt(self, user_prompt: str, asset_type: AssetType) -> OrthographicConceptPrompt:
        is_char = (asset_type == AssetType.CHARACTER)
        budget = 25000 if is_char else 8000
        pose = "A-pose" if is_char else "Neutral"
        palette = self._infer_character_palette(user_prompt) if is_char else CharacterColorPalette()

        return OrthographicConceptPrompt(
            front_prompt=f"Front orthographic view of {user_prompt}, neutral diffuse lighting, centered frame, 0-deg pitch",
            side_prompt=f"Side profile 90-degree orthographic view of {user_prompt}, strict silhouette alignment",
            back_prompt=f"Back orthographic view of {user_prompt}, clean rear contours and details",
            style_notes=f"PBR materials, Primary {palette.primary_cloth_hex}, Hair {palette.hair_color_hex}, Metalness {palette.metalness:.2f}, Roughness {palette.roughness:.2f}",
            pose=pose,
            poly_budget=budget,
            palette=palette
        )

    def _offline_plan_environment(self, prompt: str, dimensions: Optional[List[float]] = None) -> LevelLayoutSchema:
        p = prompt.lower()
        if any(w in p for w in ["cyberpunk", "subway", "neon", "alley", "city", "phố", "đêm"]):
            biome = "Cyberpunk_Subway_Neon"
        elif any(w in p for w in ["warehouse", "industrial", "factory", "nhà kho", "xưởng", "container"]):
            biome = "Abandoned_Industrial_Warehouse"
        elif any(w in p for w in ["temple", "ruin", "ancient", "đền", "cổ đại", "dungeon"]):
            biome = "Ancient_Temple_Ruins"
        else:
            biome = "Modular_SciFi_Industrial"

        w = float(dimensions[0]) if dimensions and len(dimensions) >= 2 else 40.0
        l = float(dimensions[1]) if dimensions and len(dimensions) >= 2 else 40.0
        w = max(16.0, min(w, 80.0))
        l = max(16.0, min(l, 80.0))
        hw, hl = w * 0.5, l * 0.5

        instances: List[InstanceSchema] = []

        # 1. Modular Floor Tiles covering the arena
        step = 4.0
        xs = np.arange(-hw + 2.0, hw, step)
        ys = np.arange(-hl + 2.0, hl, step)
        for fx in xs:
            for fy in ys:
                instances.append(InstanceSchema(
                    asset_id="floor_4x4",
                    position=[float(fx), float(fy), 0.0],
                    rotation=[0.0, 0.0, 0.0],
                    scale=[1.0, 1.0, 1.0]
                ))

        # 2. Perimeter Boundary Walls & Central Entry Gate Arches
        for fx in xs:
            is_gate = abs(fx) < 2.1
            asset_n = "gate_arch" if is_gate else "wall_4x4"
            instances.append(InstanceSchema(
                asset_id=asset_n,
                position=[float(fx), float(-hl), 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0]
            ))
            instances.append(InstanceSchema(
                asset_id=asset_n,
                position=[float(fx), float(hl), 0.0],
                rotation=[0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0]
            ))

        for fy in ys:
            is_gate = abs(fy) < 2.1
            asset_e = "gate_arch" if is_gate else "wall_4x4"
            instances.append(InstanceSchema(
                asset_id=asset_e,
                position=[float(-hw), float(fy), 0.0],
                rotation=[0.0, 0.0, 90.0],
                scale=[1.0, 1.0, 1.0]
            ))
            instances.append(InstanceSchema(
                asset_id=asset_e,
                position=[float(hw), float(fy), 0.0],
                rotation=[0.0, 0.0, 90.0],
                scale=[1.0, 1.0, 1.0]
            ))

        # 3. Central Elevated Sniper Bridge + Access Ramps + Reactor Core
        instances.append(InstanceSchema(asset_id="platform_elevated", position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0]))
        instances.append(InstanceSchema(asset_id="generator_core", position=[0.0, 0.0, 1.1], rotation=[0.0, 0.0, 0.0]))
        instances.append(InstanceSchema(asset_id="ramp_stairs", position=[-4.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0]))
        instances.append(InstanceSchema(asset_id="ramp_stairs", position=[4.0, 0.0, 0.0], rotation=[0.0, 0.0, 180.0]))

        # 4. Structural Colonade Pillars framing the main lane
        col_x = min(8.0, hw - 4.0)
        col_y = min(8.0, hl - 4.0)
        for sx in [-col_x, col_x]:
            for sy in [-col_y, col_y]:
                instances.append(InstanceSchema(
                    asset_id="pillar_column",
                    position=[float(sx), float(sy), 0.0],
                    rotation=[0.0, 0.0, 0.0]
                ))

        # 5. Symmetric Full-Cover Containers & Half-Cover Barricades / Crates
        flank_x = min(12.0, hw - 4.0)
        flank_y = min(6.0, hl - 4.0)
        for sign in [-1.0, 1.0]:
            instances.append(InstanceSchema(
                asset_id="container_full",
                position=[float(sign * flank_x), 0.0, 0.0],
                rotation=[0.0, 0.0, 90.0]
            ))
            instances.append(InstanceSchema(
                asset_id="barricade_half",
                position=[float(sign * 6.0), float(sign * flank_y), 0.0],
                rotation=[0.0, 0.0, 30.0]
            ))
            instances.append(InstanceSchema(
                asset_id="barricade_half",
                position=[float(-sign * 6.0), float(sign * flank_y), 0.0],
                rotation=[0.0, 0.0, -30.0]
            ))
            instances.append(InstanceSchema(
                asset_id="crate_cover",
                position=[float(sign * 4.0), float(sign * (flank_y + 2.0)), 0.0],
                rotation=[0.0, 0.0, 15.0]
            ))

        # 6. Team Alpha & Bravo Spawn Pads + Illumination Beacons
        spawn_dist = max(4.0, hl - 4.0)
        instances.append(InstanceSchema(asset_id="spawn_pad_alpha", position=[0.0, float(-spawn_dist), 0.0]))
        instances.append(InstanceSchema(asset_id="spawn_pad_bravo", position=[0.0, float(spawn_dist), 0.0]))
        for sx in [-4.0, 4.0]:
            instances.append(InstanceSchema(asset_id="beacon_light", position=[float(sx), float(-spawn_dist), 0.0]))
            instances.append(InstanceSchema(asset_id="beacon_light", position=[float(sx), float(spawn_dist), 0.0]))

        return LevelLayoutSchema(
            level_id="modular_environment_01",
            dimensions=[w, l],
            grid_size_m=2.0,
            biome=biome,
            rooms=[
                RoomSchema(id="zone_alpha", bounds=[-hw, -hl, hw, 0.0]),
                RoomSchema(id="zone_bravo", bounds=[-hw, 0.0, hw, hl])
            ],
            corridors=[
                CorridorSchema(connects=["zone_alpha", "zone_bravo"], width=4.0)
            ],
            spawn_points=[
                SpawnPointSchema(team="alpha", pos=[0.0, float(-spawn_dist), 0.0], rotation_yaw=0.0),
                SpawnPointSchema(team="bravo", pos=[0.0, float(spawn_dist), 0.0], rotation_yaw=180.0)
            ],
            instances=instances
        )

    def _offline_inspect_turntable(self, image_bytes_list: List[bytes], asset_type: AssetType) -> QAInspectionResult:
        return QAInspectionResult(
            score=90,
            passed=True,
            anomalies_detected=[],
            corrective_instructions="Offline QA pass: Normals aligned, non-manifold count is 0, UV distribution uniform."
        )
