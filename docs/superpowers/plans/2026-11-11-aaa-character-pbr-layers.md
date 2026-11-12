# AAA Character Multi-Layer & PBR Texture Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng kiến trúc nhân vật 3D chuẩn AAA gồm 3 tầng lưới tách biệt (Base Body gầy guộc, Clothing Shell nếp vải bồng bềnh, Hair Cards ruy-băng rủ sợi) kết hợp bộ texture PBR 2K (Albedo chi tiết, Tangent-space Normal map, Roughness phân vùng, Alpha Clipping).

**Architecture:**
- Module `services/inference/pbr_texture_engine.py`: Sinh bộ map PBR (Albedo, Tangent-space Normal từ ma trận độ cao qua Sobel filter, Roughness phân vùng theo chất liệu da/máu/vải, Alpha Mask cho lọn tóc và gấu váy tưa rách).
- Module `services/inference/multi_layer_character.py`: Sinh 3 sub-mesh độc lập (Base Body giải phẫu hốc hác, Clothing Shell mỏng có nếp nhăn vải bao quanh thân, Hair Cards đa tầng gồm các dải ribbon quad-strip uốn lượn) kèm tọa độ UV chuẩn.
- Đóng gói glTF 2.0 Scene với PBR Metallic-Roughness Materials và kết nối `alphaCutoff: 0.5`.
- Nâng cấp Three.js WebGL Viewer (`frontend/index.html`): Renderer hỗ trợ PBR đa kênh và Alpha Clipping tưa rách.

**Tech Stack:** Python 3.11, NumPy, SciPy, Pillow, Trimesh, FastAPI, Three.js WebGL.

**Spec:** `docs/superpowers/specs/2026-11-11-aaa-character-pbr-layers-design.md`

## Global Constraints
- **Multi-layer Separation**: Model xuất ra phải chứa các sub-mesh rõ ràng (`Body`, `Clothing`, `HairCards`), không bị dính chùm thành một khối monolithic.
- **PBR Texture Standard**: Normal map phải tuân thủ chuẩn Tangent-space OpenGL/glTF (kênh Red: $+X$, Green: $+Y$, Blue: $+Z$ với vector chuẩn hóa).
- **Alpha Clipping**: Hair cards và viền váy sử dụng Alpha Cutoff $\approx 0.5$, không bị lỗi hiển thị xuyên thấu Z-sorting.
- **Performance Budget**: Sinh hoàn chỉnh cả 3 layer lưới và bộ 4 map PBR dưới 3.0 giây trên CPU thông thường.
- **Backward Compatibility**: Endpoint `/api/v1/generate/character` giữ nguyên interface, xuất file `.glb` sẵn sàng nạp vào WebGL.

## Review Focus
- **Normal Map Orientation**: Vector pháp tuyến hướng ra ngoài bề mặt ($Z > 0$, tức kênh Blue $> 128$).
- **UV Coverage**: Tọa độ UV của tất cả các đỉnh nằm trong khoảng $[0.0, 1.0]$.
- **Alpha Masking**: Dải tóc ruy-băng có nền trong suốt ($A=0$) và sợi tóc đặc ($A=255$).
- **Cloth Separation**: Lớp vỏ váy áo không bị đâm xuyên (clipping/intersect) vào chân của Base Body.
- **Visual Verification**: Kiểm tra trực tiếp trên Three.js WebGL Studio bằng Playwright screenshot.

---

### Task 1: Xây dựng PBR Texture Engine (`services/inference/pbr_texture_engine.py`)

**Files:**
- Create: `services/inference/pbr_texture_engine.py`
- Test: `tests/test_pbr_texture_engine.py`

**Interfaces:**
- Produces:
  - `PBRTextureSet`: Dataclass chứa `albedo: Image.Image`, `normal: Image.Image`, `roughness: Image.Image`, `alpha_hair: Image.Image`, `alpha_cloth: Image.Image`.
  - `PBRTextureEngine.generate_character_textures(archetype_name: str, width: int = 1024, height: int = 1024) -> PBRTextureSet`
  - `PBRTextureEngine.height_to_normal_map(height_array: np.ndarray, strength: float = 2.0) -> Image.Image`
  - `PBRTextureEngine.generate_hair_alpha_texture(width: int = 512, height: int = 1024) -> Image.Image`
  - `PBRTextureEngine.generate_torn_cloth_alpha(width: int = 1024, height: int = 1024) -> Image.Image`

- [ ] **Step 1: Viết test cho PBRTextureEngine trong `tests/test_pbr_texture_engine.py`**
- [ ] **Step 2: Chạy test xác nhận FAIL**
- [ ] **Step 3: Cài đặt `PBRTextureEngine` trong `services/inference/pbr_texture_engine.py`**
- [ ] **Step 4: Chạy test xác nhận PASS**
- [ ] **Step 5: Commit**

---

### Task 2: Xây dựng Hair Cards Ribbon & Clothing Shell Generator

**Files:**
- Create: `services/inference/multi_layer_character.py`
- Test: `tests/test_hair_and_clothing.py`

**Interfaces:**
- Produces:
  - `generate_hair_cards(head_center: np.ndarray, num_cards: int = 40) -> trimesh.Trimesh`: Sinh dải ruy-băng cong 3 tầng với UV mapping.
  - `generate_clothing_shell(body_bounds: np.ndarray, archetype: str) -> trimesh.Trimesh`: Sinh vỏ váy áo mỏng có nếp gấp vải và UV mapping.
  - `generate_bony_base_body(archetype: str) -> trimesh.Trimesh`: Sinh thân người gầy guộc hốc hác với UV cylindrical mapping.

- [ ] **Step 1: Viết test cho Hair Cards và Clothing Shell trong `tests/test_hair_and_clothing.py`**
- [ ] **Step 2: Chạy test xác nhận FAIL**
- [ ] **Step 3: Cài đặt các hàm sinh layer trong `services/inference/multi_layer_character.py`**
- [ ] **Step 4: Chạy test xác nhận PASS**
- [ ] **Step 5: Commit**

---

### Task 3: Ghép 3 tầng lưới và tích hợp xuất glTF 2.0 PBR Material

**Files:**
- Modify: `services/inference/procedural_generator.py`
- Modify: `services/gateway.py`
- Test: `tests/test_multi_layer_character.py`

**Interfaces:**
- Modifies: `Procedural3DGenerator.generate_character_mesh` để xuất mô hình đa tầng kèm texture PBR và Alpha Clipping.

- [ ] **Step 1: Viết test kiểm tra xuất Multi-Layer GLB với vật liệu PBR trong `tests/test_multi_layer_character.py`**
- [ ] **Step 2: Chạy test xác nhận FAIL**
- [ ] **Step 3: Tích hợp `multi_layer_character` và `pbr_texture_engine` vào `procedural_generator.py`**
- [ ] **Step 4: Chạy test xác nhận PASS**
- [ ] **Step 5: Commit**

---

### Task 4: Nâng cấp WebGL Studio (`frontend/index.html`) & Kiểm thử toàn diện

**Files:**
- Modify: `frontend/index.html`
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Nâng cấp Three.js loader trong `frontend/index.html` hỗ trợ `alphaTest: 0.5`, Normal map, Roughness map, DoubleSide**
- [ ] **Step 2: Thêm thanh công cụ kiểm tra Debug Channels (Full PBR, Albedo, Normal Map, Wireframe)**
- [ ] **Step 3: Chạy toàn bộ test suite `pytest`**
- [ ] **Step 4: Khởi động server và dùng Playwright chụp ảnh thực tế kiểm tra độ chân thực của nhân vật mới**
- [ ] **Step 5: Commit hoàn thiện**
