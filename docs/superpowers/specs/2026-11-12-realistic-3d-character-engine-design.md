# Design Specification: Realistic 3D Character Generation Engine (Hunyuan3D-2 & Studio PBR)

## 1. Intent & Problem Statement
The current 3D generator in Render3D operates primarily as a procedural parametric assembler. While technically clean (watertight, fixed 25k triangle budget), the generated models appear blocky and toy-like:
- **Geometry**: Boxy voxel-like fingers, cylindrical drum-shaped chest armor, and a spherical head with flat 2D sticker facial features lacking eye sockets or nose bridge depth.
- **Neural Engine**: The public Tencent Hunyuan3D-2 Hugging Face space has direct text-to-3D disabled (`Text to 3D is disable`), causing text requests to silently fail and fall back to procedural toy models.
- **Viewport Shading**: Three.js relies on three flat colored planes rather than a true high-dynamic-range environment map (HDRI), making gold and steel look like molded ABS plastic with flat specular highlights and no material separation between organic skin and metal armor.

This specification defines the architecture to achieve **true Game-Ready Realistic 3D characters** by combining neural AI generation (Tencent Hunyuan3D-2) with studio-grade PBR rendering and organic anatomical procedural geometry.

---

## 2. Goals & Non-Goals

### Goals
1. **True AI Neural Generation (250,000+ Polygons)**:
   - Provide real neural 3D synthesis via Tencent Hunyuan3D-2 for both Text-to-3D and Image-to-3D.
   - Solve the disabled text-to-3D constraint on Hunyuan3D-2 by establishing an automated two-stage pipeline: Text -> High-resolution orthographic concept image -> Hunyuan3D-2 Image-to-3D (`/shape_generation`) -> Texture projection & export.
2. **Photorealistic PBR Material & Studio HDRI Viewport**:
   - Replace flat plane lighting with a custom HDR studio environment map generated via Three.js `PMREMGenerator` with multi-tier studio softboxes, rim lights, and ground bounce.
   - Enable `THREE.ACESFilmicToneMapping` with realistic exposure.
   - Strict material separation: Polished metals (metalness: 0.95, roughness: 0.15, crisp envMap reflection) vs. Organic skin (metalness: 0.0, roughness: 0.58, soft subsurface feel).
3. **Anatomical Upgrade to Procedural Mesh Engine (Offline/Fallback)**:
   - Hands: Articulated fingers with 3 anatomical phalanges (proximal, intermediate, distal), knuckles, and opposed thumb.
   - Head: Sculpted 3D geometry with recessed orbital cavities (eye sockets), extruded nasal bridge (cartilage ridge), and tapered jaw/chin.
   - Armor: Ergonomic curved cuirass / breastplate hugging pectoral and rib contours rather than segmented cylinders.
4. **End-to-End User Experience**:
   - Make Hunyuan3D-2 the default engine for high-detail character requests.
   - Clear generation progress indicators in UI (Concept synthesis -> Neural mesh synthesis -> Texture projection -> PBR Viewport).

### Non-Goals
- Real-time skeletal skinning on 250k neural meshes (neural models export as high-fidelity game-ready static assets or props; rigging is applied to decimated collision bodies).
- Requiring local Blender installation (system remains lightweight and standalone using pure Python, trimesh, and WebGL).

---

## 3. System Architecture & Components

```
+---------------------------------------------------------------------------------+
|                                Frontend (Three.js)                              |
|  - Studio HDRI PMREM Environment Map (Softbox + Key + Fill + Rim Lights)        |
|  - ACESFilmic Tone Mapping & High-Precision PBR Shading                         |
|  - Material Separation: Polish Metal vs. Diffuse Organic Skin                   |
+---------------------------------------------------------------------------------+
                                      |
                               (HTTP REST API)
                                      v
+---------------------------------------------------------------------------------+
|                           FastAPI Gateway (services/gateway.py)                 |
|  - Character Endpoint: Orchestrates Neural vs. Procedural Generation            |
|  - Progress Tracking & Status Logging                                           |
+---------------------------------------------------------------------------------+
           |                                                      |
           | [Engine: neural / auto]                              | [Fallback / offline]
           v                                                      v
+---------------------------------------+      +----------------------------------+
| Hunyuan3D-2 Client                    |      | Procedural Multi-Layer Generator |
| (services/inference/hunyuan3d_client) |      | (services/inference/procedural)  |
|                                       |      |                                  |
| 1. Text -> Orthographic Concept Image |      | 1. Sculpted 3D Facial Anatomy    |
|    (Front A-pose, neutral backdrop)   |      |    (Eye sockets, nose bridge)    |
| 2. Concept Image -> Hunyuan3D-2       |      | 2. Articulated 3-Segment Fingers |
|    /shape_generation (250k+ Tris)     |      | 3. Curved Ergonomic Cuirass      |
| 3. Concept UV Texture Projection      |      | 4. 2K PBR Procedural Textures    |
| 4. Export Clean GLB Asset             |      +----------------------------------+
+---------------------------------------+
```

---

## 4. Detailed Component Specifications

### 4.1 Hunyuan3D-2 Client Enhancement (`services/inference/hunyuan3d_client.py`)
- **Automated Two-Stage Text-to-3D**:
  - `generate_concept_image(prompt: str) -> str`: Uses fast AI image synthesis (e.g., Pollinations/HF) to create a clean, front orthographic A-pose character against neutral gray studio background.
  - Passes concept image using `handle_file` into `tencent/Hunyuan3D-2` `/shape_generation` space.
  - Automatically fetches the resulting `white_mesh.glb` (250,000+ polygons).
- **UV Texture Projection & PBR Baking**:
  - Projects the generated concept image onto the neural mesh geometry using front-aligned orthographic/cylindrical texture mapping.
  - Computes vertex normals and bounds alignment.
  - Exports a textured GLB ready for Three.js.

### 4.2 Three.js Viewport & Studio HDRI (`frontend/index.html`)
- **Studio Environment Lighting**:
  - Implement a programmatic Studio HDR Dome inside `setupEnvironment()` using `PMREMGenerator`.
  - Create high-contrast gradient softbox lights (warm key light at 45 deg, cool fill light, high-intensity rim light, low-intensity dark floor for specular grounding).
  - Apply `scene.environment = envTarget.texture`.
- **Tone Mapping & Specular PBR**:
  - `renderer.toneMapping = THREE.ACESFilmicToneMapping;`
  - `renderer.toneMappingExposure = 1.15;`
  - Mesh standard material settings:
    - Armor / Metallic: `metalness: 0.95`, `roughness: 0.16`, `envMapIntensity: 2.5`
    - Skin / Organic: `metalness: 0.0`, `roughness: 0.58`, `envMapIntensity: 0.5`
    - Cloth / Leather: `metalness: 0.1`, `roughness: 0.75`, `envMapIntensity: 0.8`

### 4.3 Anatomical Upgrades to Procedural Mesh Engine (`services/inference/multi_layer_character.py` & `procedural_generator.py`)
- **Articulated Hand & Fingers**:
  - Replace square voxel blocks with 5 articulated finger chains.
  - Each finger contains 3 tapered cylindrical segments (proximal, intermediate, distal) connected with spherical joints, oriented at natural resting curvature.
  - Opposed thumb positioned at 45-degree abduction.
- **3D Sculpted Facial Geometry**:
  - Head base mesh deformed with vertex displacements:
    - Depressed orbital cavities at eye coordinates ($Y \approx 1.58, X \approx \pm 0.045, Z > 0$).
    - Extruded 3D nasal bridge ($Z$ projection along midline $X=0$).
    - Defined jawline tapering into chin contour.
- **Ergonomic Armor Cuirass**:
  - Replace extruded cylinder with an anatomically curved breastplate.
  - Curvature conforms to pectoral muscle curves and rib cage tapering into the waist.
  - Prominent beveled collar flange and rim edge loops for sharp specular highlights.

---

## 5. Verification & Test Plan
1. **Neural 3D Pipeline Test**:
   - Run text prompt "A realistic golden knight in gleaming plate armor" -> verify concept image synthesis -> verify Hunyuan3D-2 generates 250k+ triangle neural mesh -> verify valid GLB export.
2. **PBR Shading Test**:
   - Inspect Three.js canvas in browser -> verify metallic specular reflections from studio softbox lights -> verify distinct roughness between skin and armor.
3. **Procedural Geometry Test**:
   - Run multi-layer procedural generation -> verify finger phalange count (3 per finger) -> verify facial 3D depth on head mesh (orbital recess & nasal bridge).
4. **Integration Test**:
   - Execute pytest test suite on gateway, hunyuan3d client, and procedural engine.
