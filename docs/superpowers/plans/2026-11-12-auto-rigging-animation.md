# Auto-Rigging & Skeletal Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement automated Humanoid Auto-Rigging & Animation for 3D character models in Render3D AI Studio, exporting multi-action animated GLBs (Walk [default], Idle, Run, Attack) viewable in Three.js with full playback controls.

**Architecture:** 
1. **Motion Library & Blender Headless**: Standard motion directory `assets/animations/` and headless script `scripts/auto_rig_and_animate.py` with bounding box armature fitting, automatic weight painting, and NLA track retargeting.
2. **Standalone Skeletal Rigging Engine**: In `services/dcc/character_rigging_engine.py`, provides zero-dependency glTF skinning & keyframed humanoid motion synthesis (Walk, Idle, Run, Attack) when Blender is not installed.
3. **Backend API Endpoint**: `POST /api/v1/animate-character` (and `/api/animate-character`) in `services/gateway.py`.
4. **Frontend Animation Controller**: Three.js `AnimationMixer` in `frontend/index.html` with action switcher (Walk as default), speed slider, play/pause, and smooth cross-fading.

**Tech Stack:** Python 3.11, Blender Python API (`bpy`), trimesh, numpy, Three.js (r128), FastAPI.

**Spec:** `docs/superpowers/specs/2026-11-12-auto-rigging-animation-design.md`

## Global Constraints
- Must function in both environments: when local Blender executable is present, and when Blender is absent (headless standalone mode).
- All generated animated models must be valid glTF binary (`.glb`) files containing humanoid bone hierarchies and animation clips.
- Default animation played in viewport must be `Walk`.

## Review Focus
1. Bounding box height normalization: models of arbitrary scale must be normalized to standard height (~1.75m) with feet on ground ($Y=0$).
2. Missing motion preset files: system must synthesize standard cyclic keyframe tracks if external FBX/BVH files are not found.
3. Animation cross-fading glitch: switching actions in Three.js must use `crossFadeTo` without mesh distortion or sudden snaps.
4. GLB node hierarchy: armature root and skin nodes must be properly linked so mesh transforms with bone rotations.
5. Play/Pause toggle and speed slider: must update mixer timeScale dynamically without restarting the clip.

---

### Task 1: Motion Library Structure & Blender Headless Script

**Files:**
- Create: `assets/animations/` (directory)
- Create: `scripts/auto_rig_and_animate.py`
- Test: `tests/test_auto_rig_script.py`

**Interfaces:**
- Produces: CLI script runnable via `blender --background --python scripts/auto_rig_and_animate.py -- <input_glb> <output_glb> [anim_names...]`
- Produces: `assets/animations/` directory with sample animation placeholders.

- [ ] **Step 1: Write test for script generation and CLI parameter parsing**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `scripts/auto_rig_and_animate.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 2: Standalone glTF Skeletal Rigging & Animation Engine Fallback

**Files:**
- Modify: `services/dcc/character_rigging_engine.py`
- Create: `tests/test_standalone_animation_engine.py`

**Interfaces:**
- Produces: `CharacterRiggingEngine.rig_and_animate_character(input_glb: str, output_glb: str, animations: List[str]) -> str`
- Binds standard Humanoid skeleton (`Hips`, `Spine`, `Chest`, `Neck`, `Head`, `LeftUpperArm`, `LeftLowerArm`, `LeftHand`, `RightUpperArm`, `RightLowerArm`, `RightHand`, `LeftUpperLeg`, `LeftLowerLeg`, `LeftFoot`, `RightUpperLeg`, `RightLowerLeg`, `RightFoot`).
- Synthesizes smooth cyclic keyframe tracks:
  - `Walk`: Alternating leg stride, hip bob, arm swing (1.0s loop).
  - `Idle`: Subtle spine breathing, slight arm sway (2.0s loop).
  - `Run`: Faster leg stride (0.6s loop) with deeper forward lean.
  - `Attack`: Torso twist and dominant arm forward slash (1.2s action).

- [ ] **Step 1: Write failing test for `rig_and_animate_character`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement standalone glTF humanoid skinning and animation track builder in `character_rigging_engine.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 3: Backend API Endpoint `/api/v1/animate-character`

**Files:**
- Modify: `services/gateway.py`
- Modify: `tests/test_gateway.py`

**Interfaces:**
- Consumes: `CharacterRiggingEngine.rig_and_animate_character`, `scripts/auto_rig_and_animate.py`
- Produces: `POST /api/v1/animate-character` (and `/api/animate-character`) endpoint returning `{ "status": "completed", "model_url": "...", "animations": [...] }`

- [ ] **Step 1: Write test for animate character endpoint**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement endpoint in `services/gateway.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 4: Three.js Frontend Animation Controls & Playback

**Files:**
- Modify: `frontend/index.html`

**Interfaces:**
- Consumes: `THREE.AnimationMixer`, `THREE.Clock`
- Produces: UI action selector (Walk [Default], Idle, Run, Attack, T-Pose), Play/Pause button, speed slider (0.5x - 2.0x).
- Auto-triggers animation upon model generation or rigging.

- [ ] **Step 1: Add animation toolbar controls to `frontend/index.html`**
- [ ] **Step 2: Integrate `THREE.AnimationMixer` with `crossFadeTo` in `loadModel()`**
- [ ] **Step 3: Set `Walk` as default active clip on load**
- [ ] **Step 4: Verify in test suite**
- [ ] **Step 5: Commit**
