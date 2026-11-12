# Realistic 3D Character Generation Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Render3D character generation from toy-like blocky procedural models to photorealistic, game-ready 3D characters via Tencent Hunyuan3D-2 neural generation (250k+ faces), realistic Three.js Studio HDRI lighting, and anatomical procedural geometry.

**Architecture:** A dual-engine pipeline:
1. **Neural AI Engine (Primary)**: Automated 2-stage bridge (Text -> Orthographic A-pose Concept Image -> Hunyuan3D-2 `/shape_generation` -> UV texture projection) yielding 250k+ polygon detailed neural meshes.
2. **Three.js Studio Viewport**: Programmatic Studio HDRI environment via PMREMGenerator, ACESFilmic tone mapping, and strict PBR material parameter separation (polished metal vs. diffuse organic skin).
3. **Anatomical Procedural Engine (Fallback/Offline)**: 3-segment articulated finger chains, 3D sculpted orbital cavities and nasal bridge on head geometry, and ergonomic curved cuirass armor.

**Tech Stack:** Python 3.11, FastAPI, gradio_client, trimesh, Pillow, numpy, scipy, Three.js (r128), WebGL.

**Spec:** `docs/superpowers/specs/2026-11-12-realistic-3d-character-engine-design.md`

## Global Constraints
- Python 3.11 compatibility; all external dependencies must use standard libraries or existing requirements (trimesh, Pillow, numpy, scipy, gradio_client).
- No requirement for local Blender executable (`blender_available: False` must be fully supported).
- All generated meshes must be valid, manifold, watertight GLB files viewable in standard Three.js GLTFLoader.
- Three.js viewport must run in-browser without external network CDN dependencies beyond current imports.

## Review Focus
1. Hunyuan3D space `/shape_generation` API connection failure or timeout: must log clearly and gracefully fall back to upgraded anatomical procedural engine.
2. Concept image URL fetch failure (e.g. offline): must generate a local procedural concept image fallback before calling Hunyuan3D.
3. Neural mesh scale and coordinate orientation: Hunyuan3D GLBs must be normalized to standard height (~1.8m) and Y-up orientation before Three.js load.
4. Procedural hand mesh vertex explosion or degenerate faces: 3-segment phalanges must maintain manifold connectivity with the palm.
5. Three.js environment map memory leak: PMREM texture generation must properly dispose of intermediate render targets.

---

### Task 1: Automated 2-Stage Text-to-3D & UV Texture Projection in Hunyuan3D Client

**Files:**
- Modify: `services/inference/hunyuan3d_client.py`
- Create: `tests/test_hunyuan3d_neural_flow.py`

**Interfaces:**
- Consumes: `HF_TOKEN` from `.env`, `gradio_client.Client`, `gradio_client.handle_file`
- Produces: `Hunyuan3DClient.generate_concept_image(caption: str, output_path: str) -> str`
- Produces: `Hunyuan3DClient.generate_3d_from_text(caption: str, output_path: str, steps: int = 15, timeout: int = 180) -> str`
- Produces: `Hunyuan3DClient.project_texture_to_mesh(glb_path: str, image_path: str, output_glb_path: str) -> str`

- [ ] **Step 1: Write the failing test for 2-stage text-to-3d and texture projection**

```python
# tests/test_hunyuan3d_neural_flow.py
import pytest
from pathlib import Path
from services.inference.hunyuan3d_client import Hunyuan3DClient

def test_concept_image_generation_fallback(tmp_path):
    client = Hunyuan3DClient()
    out_img = str(tmp_path / "concept.png")
    res_path = client.generate_concept_image("realistic knight in golden armor", out_img)
    assert Path(res_path).exists()
    assert Path(res_path).stat().st_size > 1000

def test_project_texture_to_mesh_creates_material(tmp_path):
    import trimesh
    client = Hunyuan3DClient()
    # Create simple box mesh
    box = trimesh.creation.box(extents=(1, 2, 0.5))
    raw_glb = str(tmp_path / "raw.glb")
    box.export(raw_glb)
    
    # Generate test image
    img_path = str(tmp_path / "concept.png")
    client.generate_concept_image("knight", img_path)
    
    out_glb = str(tmp_path / "textured.glb")
    client.project_texture_to_mesh(raw_glb, img_path, out_glb)
    
    assert Path(out_glb).exists()
    scene = trimesh.load(out_glb)
    geom = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    assert hasattr(geom.visual, "uv") and geom.visual.uv is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hunyuan3d_neural_flow.py -v`
Expected: FAIL with `AttributeError: 'Hunyuan3DClient' object has no attribute 'generate_concept_image'`

- [ ] **Step 3: Implement `generate_concept_image`, `project_texture_to_mesh`, and update `generate_3d_from_text` in `services/inference/hunyuan3d_client.py`**

- `generate_concept_image`: Fetches high-res orthographic A-pose render from Pollinations AI (with neutral studio backdrop), falls back to procedural front view if offline.
- `project_texture_to_mesh`: Loads mesh via trimesh, normalizes Y-up and centered bounds, generates cylindrical/planar UVs along bounding box front projection, binds image as texture material, exports textured GLB.
- `generate_3d_from_text`: Generates concept image -> submits to Hunyuan3D `/shape_generation` with `handle_file` -> projects concept image onto resulting mesh -> writes final textured GLB.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hunyuan3d_neural_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/inference/hunyuan3d_client.py tests/test_hunyuan3d_neural_flow.py
git commit -m "feat: implement 2-stage text-to-3d and uv texture projection in hunyuan3d client"
```

---

### Task 2: Three.js Viewport Studio HDRI Environment & Realistic PBR Shading

**Files:**
- Modify: `frontend/index.html`

**Interfaces:**
- Consumes: Three.js `PMREMGenerator`, `THREE.ACESFilmicToneMapping`
- Produces: `setupEnvironment()` with dynamic studio softbox HDR map
- Produces: `loadModel()` with differentiated PBR properties (`metalness`, `roughness`, `envMapIntensity`)

- [ ] **Step 1: Update `setupEnvironment()` in `frontend/index.html`**

Replace 3 flat planes with a comprehensive studio lighting rig in `envScene`:
- Key softbox at 45 deg (intensity 3.0, warm white #fff5ea)
- Fill softbox at -45 deg (intensity 1.2, cool #dbeafe)
- High-intensity rim light at top rear (intensity 4.0, #ffffff)
- Dark ground plane with subtle specular floor bounce (#111827)
- Generate environment map via `pmremGenerator.fromScene(envScene, 0.04)` and assign to `scene.environment`.
- Configure renderer: `renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.15;`.

- [ ] **Step 2: Update `loadModel()` material parser in `frontend/index.html`**

Ensure materials on incoming meshes are parsed and enhanced:
- Metal/Armor detection: `child.material.metalness = 0.95; child.material.roughness = 0.16; child.material.envMapIntensity = 2.4;`.
- Skin detection: `child.material.metalness = 0.0; child.material.roughness = 0.58; child.material.envMapIntensity = 0.5;`.
- Cloth/Leather detection: `child.material.metalness = 0.08; child.material.roughness = 0.72; child.material.envMapIntensity = 0.8;`.
- Set default engine dropdown to `neural` (Tencent Hunyuan3D-2).

- [ ] **Step 3: Verify frontend markup and script in browser / test server**

Run: `python -m pytest tests/test_gateway.py`
Verify no syntax or loading errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html
git commit -m "feat: add studio hdri environment, acesfilmic tone mapping, and pbr material separation to frontend"
```

---

### Task 3: Anatomical Upgrades to Procedural Mesh Engine (Hands, Face, Cuirass)

**Files:**
- Modify: `services/inference/multi_layer_character.py`
- Modify: `services/inference/procedural_generator.py`
- Create: `tests/test_anatomical_procedural_mesh.py`

**Interfaces:**
- Produces: `_build_anatomical_hand(is_left: bool) -> trimesh.Trimesh` with 3-segment articulated phalanges (proximal, intermediate, distal) per finger.
- Produces: `_sculpt_facial_features(head_mesh: trimesh.Trimesh) -> trimesh.Trimesh` with 3D indented orbital cavities and extruded nasal ridge.
- Produces: `_build_curved_cuirass() -> trimesh.Trimesh` with curved pectoral contour and waist tapering.

- [ ] **Step 1: Write failing tests for anatomical hand, sculpted head, and curved cuirass**

```python
# tests/test_anatomical_procedural_mesh.py
import pytest
import numpy as np
import trimesh
from services.inference.multi_layer_character import MultiLayerCharacterGenerator

def test_hand_has_articulated_finger_segments():
    gen = MultiLayerCharacterGenerator()
    hand = gen._build_anatomical_hand(is_left=False)
    assert len(hand.vertices) > 200
    assert len(hand.faces) > 300
    # Hand length should span from wrist to fingertips
    bounds = hand.bounds
    length = bounds[1][1] - bounds[0][1]
    assert 0.12 <= length <= 0.25

def test_head_mesh_has_sculpted_eye_sockets_and_nose():
    gen = MultiLayerCharacterGenerator()
    head = gen._build_sculpted_head()
    # Check that head vertices in the eye region have indented Z compared to forehead/cheeks
    eye_y_range = (1.54, 1.62)
    verts = head.vertices
    eye_verts = verts[(verts[:, 1] >= eye_y_range[0]) & (verts[:, 1] <= eye_y_range[1]) & (verts[:, 2] > 0)]
    assert len(eye_verts) > 0
    # Nose ridge vertices should have higher Z than eye sockets
    nose_verts = eye_verts[np.abs(eye_verts[:, 0]) < 0.015]
    socket_verts = eye_verts[np.abs(eye_verts[:, 0]) > 0.035]
    if len(nose_verts) > 0 and len(socket_verts) > 0:
        assert np.max(nose_verts[:, 2]) > np.min(socket_verts[:, 2])

def test_cuirass_has_curved_pectoral_geometry():
    gen = MultiLayerCharacterGenerator()
    cuirass = gen._build_curved_cuirass()
    assert len(cuirass.faces) > 400
    # Should not be a flat cylinder
    bounds = cuirass.bounds
    width = bounds[1][0] - bounds[0][0]
    depth = bounds[1][2] - bounds[0][2]
    assert width > depth * 1.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_anatomical_procedural_mesh.py -v`
Expected: FAIL with `AttributeError: 'MultiLayerCharacterGenerator' object has no attribute '_build_anatomical_hand'`

- [ ] **Step 3: Implement anatomical hand, sculpted head, and curved cuirass in `multi_layer_character.py` and integrate into `procedural_generator.py`**

- `_build_anatomical_hand`: Generates palm base + 4 fingers (each with 3 cylindrical phalanges connected with spherical knuckles) + 1 opposed thumb rotated 45 degrees.
- `_build_sculpted_head`: Generates UV sphere or icosphere base, calculates vertex distances to left/right orbital centers ($X=\pm 0.045, Y=1.58, Z=0.08$) and depresses vertices inward ($Z$ reduced by Gaussian weight); extrudes nasal ridge vertices along midline ($X \approx 0, Y \in [1.53, 1.60]$) outward in $+Z$.
- `_build_curved_cuirass`: Generates anatomical chest plate with convex pectoral bulges, waist taper, and beveled neck collar rim.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_anatomical_procedural_mesh.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/inference/multi_layer_character.py services/inference/procedural_generator.py tests/test_anatomical_procedural_mesh.py
git commit -m "feat: upgrade procedural mesh engine with articulated fingers, sculpted facial anatomy, and curved cuirass"
```

---

### Task 4: Gateway Routing & End-to-End Integration

**Files:**
- Modify: `services/gateway.py`
- Modify: `tests/test_gateway.py`

**Interfaces:**
- Consumes: `Hunyuan3DClient.generate_3d_from_text`, `Procedural3DGenerator.generate_multi_layer_character`
- Produces: Updated `/api/v1/generate/character` endpoint with smart neural prioritization and detailed telemetry.

- [ ] **Step 1: Write integration test for character generation endpoint**

```python
# In tests/test_gateway.py
def test_generate_character_neural_or_fallback(client):
    res = client.post("/api/v1/generate/character", json={
        "prompt": "A realistic golden knight in gleaming plate armor",
        "style": "Game-Ready Realistic",
        "poly_budget": 25000,
        "engine": "neural"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert "model_url" in data
    assert Path(f"storage/assets/{Path(data['model_url']).name}").exists()
```

- [ ] **Step 2: Update `services/gateway.py`**

- Enhance `/api/v1/generate/character`: If `engine == "neural"` or `engine == "auto"`:
  - Attempt Hunyuan3D-2 2-stage text-to-3d synthesis.
  - If successful, return neural asset with `engine: "tencent/Hunyuan3D-2 (Neural)"` and QA score $\ge 96$.
  - If remote queue fails or token unavailable, fall back to upgraded anatomical procedural studio with clear logging and status note.

- [ ] **Step 3: Run all test suites**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add services/gateway.py tests/test_gateway.py
git commit -m "feat: wire gateway character endpoint to 2-stage neural pipeline with anatomical fallback"
```
