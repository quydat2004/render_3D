# Seamless Character Hybrid 3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuyển đổi bộ sinh nhân vật procedural từ ghép khối rời rạc sang lưới đơn nhất liền mạch 100% (Single Watertight Manifold Mesh) bằng giải tích SDF Smooth Blending và Marching Cubes, tương thích với DCC Remeshing.

**Architecture:** Sử dụng toán tử Signed Distance Field (SDF) và hàm hòa trộn đa thức trơn ($smin$) để nối liền tự nhiên các vùng cơ thể (cổ, vai, nách, khuỷu tay, ngực, chân). Trích xuất bề mặt kín bằng `scikit-image.measure.marching_cubes`, gán vertex colors PBR theo vị trí giải phẫu, chuyển đổi Y-up, và đưa qua Blender QuadriFlow/Trimesh post-process.

**Tech Stack:** Python 3.11, NumPy, SciPy, scikit-image, Trimesh, FastAPI, Three.js WebGL.

**Spec:** `docs/superpowers/specs/2026-11-11-seamless-character-hybrid-design.md`

## Global Constraints
- Single Watertight Mesh: Mesh nhân vật đầu ra phải là một manifold thống nhất, không có khối giao nhau rời rạc bên trong.
- Performance: Thời gian sinh lưới nhân vật trên CPU dưới 1.0 giây.
- API Contract: Giữ nguyên interface của `generate_character_mesh(prompt, concept)` trả về `trimesh.Trimesh`.
- Trục tọa độ: Định dạng đầu ra cuối cùng phải tuân thủ chuẩn Three.js / glTF (Y-up, chân chạm sàn tại Y=0).

## Review Focus
- Bề mặt kín không thủng (watertight/manifold): Kiểm tra qua `mesh.is_watertight` hoặc `len(mesh.faces) > 0` và số lượng thành phần liên thông bằng 1 (`len(mesh.split(only_watertight=False)) == 1`).
- Landmark rigging tương thích: Kiểm tra `char_rigger.estimate_landmarks(mesh)` nhận diện đúng đầy đủ 16 khớp cơ thể.
- Đầy đủ màu sắc PBR (vertex colors): Đảm bảo các đỉnh mang mảng RGBA hợp lệ.
- Xử lý các archetype khác nhau: `WARRIOR_HUMANOID`, `GHOST_ROBED`, `MECHA_HEAVY` đều sinh ra mesh hợp lệ.
- Fallback khi không có Blender: Tự động chạy mượt mà trên Trimesh.

---

### Task 1: Bổ sung dependencies vào requirements.txt

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `scikit-image>=0.22.0`, `gradio_client>=2.7.0`, `python-multipart>=0.0.9`

- [ ] **Step 1: Cập nhật `requirements.txt` với các thư viện cần thiết**
- [ ] **Step 2: Kiểm tra import thành công trong Python**
- [ ] **Step 3: Chạy test config `pytest tests/test_config.py`**
- [ ] **Step 4: Commit thay đổi**

---

### Task 2: Cài đặt toán tử SDF cơ sở và Smooth Blending trong `procedural_generator.py`

**Files:**
- Modify: `services/inference/procedural_generator.py`
- Test: `tests/test_procedural_generator.py`

**Interfaces:**
- Produces:
  - `_smin(d1, d2, k=0.08)`: Hàm hòa trộn trơn
  - `_sdf_sphere(P, center, radius)`: SDF quả cầu (đầu, khớp)
  - `_sdf_capsule(P, a, b, radius)`: SDF hình con nhộng đoạn thẳng (chi, cổ, thân)
  - `_sdf_ellipsoid(P, center, radii)`: SDF hình phỏng cầu elip (ngực, tóc)
  - `_sdf_box(P, center, extents)`: SDF hình hộp

- [ ] **Step 1: Viết test kiểm tra các hàm toán tử SDF trong `tests/test_procedural_generator.py`**
- [ ] **Step 2: Chạy test để xác nhận fails**
- [ ] **Step 3: Cài đặt các hàm toán tử SDF vector hóa trên NumPy trong `services/inference/procedural_generator.py`**
- [ ] **Step 4: Chạy lại test xác nhận PASS**
- [ ] **Step 5: Commit**

---

### Task 3: Tái cấu trúc `generate_character_mesh` với Seamless SDF & Marching Cubes

**Files:**
- Modify: `services/inference/procedural_generator.py`
- Test: `tests/test_procedural_generator.py`

**Interfaces:**
- Modifies: `Procedural3DGenerator.generate_character_mesh(prompt: str, concept=None) -> trimesh.Trimesh`

- [ ] **Step 1: Viết test kiểm tra tính liền mạch (single connected component, vertex colors, Y-up) trong `tests/test_procedural_generator.py`**
- [ ] **Step 2: Cài đặt hàm đánh giá trường thể tích SDF toàn thân nhân vật (Head, Neck, Torso, Arms in A-pose, Legs/Gown, Face Features)**
- [ ] **Step 3: Trích xuất bề mặt bằng `skimage.measure.marching_cubes`**
- [ ] **Step 4: Gán vertex colors PBR theo vị trí không gian giải phẫu**
- [ ] **Step 5: Chuyển đổi Z-up sang Y-up và tiếp sàn tại Y=0**
- [ ] **Step 6: Chạy test `pytest tests/test_procedural_generator.py` xác nhận PASS**
- [ ] **Step 7: Commit**

---

### Task 4: Kiểm thử tích hợp toàn diện hệ thống (Rigging, Gateway API, DCC Pipeline)

**Files:**
- Test: `tests/test_character_rigging.py`
- Test: `tests/test_blender_pipeline.py`
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Chạy test rigging với mesh nhân vật liền mạch mới**
- [ ] **Step 2: Chạy test blender pipeline processor**
- [ ] **Step 3: Chạy test gateway endpoint `/api/v1/generate/character`**
- [ ] **Step 4: Chạy toàn bộ test suite `pytest`**
- [ ] **Step 5: Commit hoàn thiện**
