# Design: AAA Multi-Layered 3D Character Architecture & Hybrid PBR Texture Engine

## 1. Intent & The Why
Hiện tại, nhân vật 3D được tạo thành một khối liền (single monolithic mesh) và dùng vertex colors. Để đạt chuẩn đồ họa chân thực, điện ảnh và kinh dị cao cấp (tương tự Silent Hill, Resident Evil, Black Myth: Wukong), nhân vật không thể là một khối đặc mà phải được cấu thành từ 3 lớp hình học độc lập (Multi-Layered Architecture):
1. **Base Body**: Giải phẫu người gầy guộc (xương sườn, xương quai xanh, hốc mắt sâu, ngón tay dài nhọn).
2. **Hair Cards**: Các dải đa giác mỏng uốn cong rủ từ đỉnh đầu xuống vai, dùng Alpha Clipping hiển thị từng sợi tóc tự nhiên.
3. **Clothing Shell**: Lớp áo váy mỏng phủ ngoài cơ thể, có nếp gấp vải và gấu váy tưa rách qua Alpha Mask.
Kèm theo đó là **Bộ texture PBR đa kênh (2048x2048)** gồm Albedo, Normal Map (tạo độ gồ ghề của gân máu và sợi vải), Roughness Map (phân biệt da ướt/máu bóng loáng với vải thô khô khốc), AO Map (đổ bóng hốc mắt, nếp vải), và Alpha Mask.

---

## 2. Goals, Non-Goals, Anti-Goals

### Goals
- **Multi-layered Mesh Architecture**:
  - Xuất file `.glb` chứa đầy đủ 3 sub-mesh / primitives: `Mesh_Body`, `Mesh_Clothing`, `Mesh_HairCards`.
  - `Mesh_Body`: Dáng vóc giải phẫu chi tiết, có xương quai xanh, lồng ngực gân guốc, hốc mắt sâu, ngón tay thon dài.
  - `Mesh_Clothing`: Vỏ lưới tách rời bao ngoài thân và xòe thành tà váy có nếp nhăn lượn sóng tự nhiên.
  - `Mesh_HairCards`: Hệ thống dải đa giác uốn lượn (curved ribbon strips) phân 3 tầng (lớp bám sát đầu, lớp tạo độ dày volume, và lớp lọn sợi rủ trước ngực/sau lưng).
- **Hybrid PBR 2K Texture Engine (`services/inference/pbr_texture_engine.py`)**:
  - Sinh bộ map 2048x2048 (hoặc 1024x1024 tối ưu):
    - `albedo.png`: Tông da bợt, vết thâm quanh mắt, máu khô thẫm ở ngực/váy, bụi bẩn lem ở chân váy.
    - `normal.png`: Tangent-space normal map sinh từ ma trận độ cao (Sobel gradient filter) tạo nổi gân máu, sống mũi, nếp nhăn vải.
    - `roughness_metallic.png`: Kênh Roughness (G) phân biệt máu/mắt bóng bẩy ($0.1-0.2$) với da ($0.5$) và vải thô ($0.85$).
    - `alpha_mask.png`: Kênh Alpha cắt tỉa sợi tóc và viền váy răng cưa tưa tả.
- **UV Coordinate Mapping**:
  - Tự động gán UV tọa độ cylindrical / planar chuẩn cho từng layer để texture áp khớp hoàn hảo.
- **Frontend 3D Viewport Three.js**:
  - Bật `alphaTest: 0.5`, `transparent: true`, `side: THREE.DoubleSide` cho vật liệu tóc và váy.
  - Tích hợp thanh chuyển kênh kiểm tra (Debug Channels: Full PBR, Albedo, Normal, Roughness, Wireframe).

### Non-Goals
- Không thay đổi các endpoint tạo vũ khí hay bản đồ môi trường.
- Không yêu cầu card GPU rời (thuần CPU qua NumPy, SciPy, Pillow, Trimesh, đảm bảo chạy ổn định trên mọi môi trường máy chủ).

### Anti-Goals
- Không quay lại kiến trúc một khối monolithic đục đặc không có layer.
- Không dùng texture độ phân giải thấp gây vỡ hạt.

---

## 3. Kiến trúc Kỹ thuật Chi tiết

### 3.1. Phân tầng hình học (`MultiLayerCharacterGenerator`)
1. **Base Body**:
   - Sử dụng giải tích SDF sinh ra thân người gầy guộc, cổ, mặt hốc hác, xương quai xanh, chi tay chân.
   - UV mapping: Cylindrical unwrap chiếu tọa độ $U = \text{atan2}(x, y) / 2\pi + 0.5$, $V = (z - z_{\min}) / (z_{\max} - z_{\min})$.
2. **Clothing Geometry (Vỏ váy áo)**:
   - Sinh lớp vỏ mỏng bọc quanh thân từ ngực ($Z=1.35$) xuống qua đầu gối ($Z=0.20$).
   - Bổ sung hàm biến thiên sóng điều hòa $\sin(8\theta)$ tạo nếp gấp vải bồng bềnh tự nhiên, bán kính mở rộng cách chân $3-6\text{ cm}$ để không bị chìm vào da.
3. **Hair Cards (Lưới dải tóc cong)**:
   - Mỗi card là một dải ruy-băng gồm $N$ mắt đốt (quad strip) uốn lượn theo đường cong Spline/Bézier từ chân tóc trên đỉnh đầu rủ xuống vai, ngực và lưng.
   - UV gán dọc theo chiều dài dải ruy-băng $[0, 1] \times [0, 1]$ để map đúng lọn tóc alpha.

### 3.2. PBR Texture Generator
- **Albedo Map**:
  - Lớp nền màu da nhợt nhạt xác ướp kết hợp gradient độ cao.
  - Vùng mắt: Hốc thâm tím đậm (bruises quanh mắt).
  - Vùng miệng: Vết son/máu thẫm loang.
  - Vùng ngực & váy: Vết máu khô đỏ sẫm loang theo vệt nhỏ giọt.
  - Chân váy: Bụi đất sẫm màu.
- **Normal Map (Tangent-Space)**:
  - Sinh từ Height Map: $N_x = -\partial H / \partial x$, $N_y = -\partial H / \partial y$, $N_z = 1.0$.
  - Chuẩn hóa vector: $\vec{N} = \text{normalize}([N_x, N_y, N_z])$.
  - Mã hóa vào ảnh RGB: $\text{Color} = (\vec{N} \times 0.5 + 0.5) \times 255$.
- **Roughness Map**:
  - Mắt & máu: $0.15$ (rất bóng, ướt át).
  - Da mặt & thân: $0.50$.
  - Vải áo: $0.85$ (mờ thô ráp).
- **Alpha Mask**:
  - Dải tóc: Vẽ nhiều sợi tóc đan xen, làm mờ cạnh, nền trong suốt hoàn toàn ($A=0$).
  - Gấu váy: Nhiễu Perlin/răng cưa ngẫu nhiên cắt đứt viền tà váy tạo hiệu ứng rách rưới.

### 3.3. Tích hợp glTF 2.0 & Three.js
- Export ra file `.glb` với PBR material chứa:
  - `pbrMetallicRoughness.baseColorTexture`
  - `normalTexture`
  - `pbrMetallicRoughness.metallicRoughnessTexture`
  - Material cho Hair và Cloth bật `alphaMode: "MASK"`, `alphaCutoff: 0.5`.
- WebGL Frontend nạp model hiển thị tức thì với ánh sáng PBR Studio HDRI.

---

## 4. Kế hoạch Kiểm thử (Test Plan)
1. `tests/test_pbr_texture_engine.py`:
   - Kiểm tra sinh đủ 4 map (Albedo, Normal, Roughness, Alpha) đúng kích thước $1024 \times 1024$ / $2048 \times 2048$.
   - Kiểm tra Normal Map có vector dương $Z$ (kênh Blue $> 128$).
2. `tests/test_multi_layer_character.py`:
   - Kiểm tra mesh sinh ra có đủ 3 layers phân biệt: Body, Clothing, Hair Cards.
   - Kiểm tra Hair Cards có tọa độ UV $U, V \in [0, 1]$.
3. `tests/test_gateway.py`:
   - Endpoint `/api/v1/generate/character` trả về file `.glb` chứa đầy đủ multi-layer mesh và vật liệu PBR.
4. Kiểm tra trực tiếp trên Three.js Web UI bằng Playwright screenshot.
