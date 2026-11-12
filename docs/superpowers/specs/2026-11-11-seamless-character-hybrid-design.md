# Design: Seamless Organic Character 3D Generator via Hybrid SDF & DCC Remeshing

## 1. Intent & The Why
Hiện tại, `Procedural3DGenerator` tạo nhân vật bằng cách nối các khối rời (icosphere, box, cylinder) bằng `trimesh.util.concatenate`. Các khối này không có sự gắn kết về mặt topo học (geometry topology), lộ rõ các đường cắt, ngấn và khe hở tại khớp cổ, vai, nách, khuỷu tay, háng, gối.
Mục tiêu là chuyển đổi sang mô hình **Single Watertight Manifold Mesh (Lưới đơn nhất, liền mạch 100%)** bằng cơ chế **Hybrid (SDF Voxel Smooth Blending + DCC Remeshing)** để nhân vật có hình thể giải phẫu mềm mại, cơ bắp và khớp nối tự nhiên, độ chi tiết cao.

## 2. Goals, Non-Goals, Anti-Goals

### Goals
- Tạo ra lưới 3D nhân vật đơn nhất (single watertight manifold mesh) không còn các mặt cắt bên trong (internal faces) hay khe hở khớp nối.
- Sử dụng hàm giải tích khoảng cách có dấu (Signed Distance Field - SDF) kết hợp hàm hòa trộn trơn (`smooth minimum` / $smin$) để nối liền tự nhiên giữa:
  - Đầu, mặt, cằm với cổ.
  - Cổ với ngực và bả vai.
  - Vai với bắp tay, khuỷu tay và cẳng tay.
  - Thân trên với hông và chân.
- Trích xuất lưới đa giác bằng thuật toán Marching Cubes trên lưới thể tích (voxel grid) độ phân giải thích hợp để bảo toàn chi tiết cơ thể và đặc điểm khuôn mặt (mắt, mũi, môi).
- Gán màu sắc PBR (vertex colors) mượt mà dựa trên vị trí giải phẫu (da, tóc, áo giáp, trang phục).
- Hỗ trợ cả 2 pipeline:
  - **DCC Mode (Blender)**: Chạy QuadriFlow remesh và UV Smart Project.
  - **Fallback Mode (Trimesh / SciPy)**: Tự động chạy Laplacian smoothing và tính toán chuẩn pháp tuyến (smooth vertex normals).

### Non-Goals
- Không thay đổi API contract của `/api/v1/generate/character` hay định dạng trả về (giữ nguyên cấu trúc JSON và URL tải `.glb`).
- Không làm thay đổi cơ chế sinh vũ khí hoặc bản đồ môi trường.

### Anti-Goals
- Không sử dụng thư viện C++ ngoài đòi hỏi biên dịch phức tạp hoặc GPU độc quyền (giữ mã nguồn chạy thuần CPU qua NumPy/SciPy/scikit-image/trimesh để bảo đảm tính portability trên mọi máy).
- Không chấp nhận thời gian sinh lưới vượt quá 3-5 giây trên CPU thông thường.

## 3. Constraints & Technical Architecture

### 3.1. SDF Math Primitives & Smooth Minimum
- Các toán tử cơ sở:
  - `sdf_sphere(p, r)`
  - `sdf_capsule(p, a, b, r)` (rất phù hợp cho tay, chân, ngón, cổ)
  - `sdf_ellipsoid(p, r)` (phù hợp cho ngực, đầu)
  - `sdf_box(p, b)`
- Hàm hòa trộn trơn mượt hữu cơ (Polynomial smooth minimum):
  $$smin(a, b, k) = \min(a, b) - \frac{\max(k - |a - b|, 0)^2}{4k}$$
  với tham số $k$ điều chỉnh độ mềm mại tại khớp nối (ví dụ $k=0.04$ đến $0.08$).

### 3.2. Voxel Marching Cubes
- Thư viện: `scikit-image.measure.marching_cubes` (đã bổ sung vào `requirements.txt`).
- Không gian tính toán voxel: Bounding box $X \in [-0.6, 0.6]$, $Y \in [-0.4, 0.4]$, $Z \in [0.0, 1.95]$.
- Độ phân giải voxel: $80 \times 60 \times 130$ (~624.000 điểm mẫu, tính toán trong ~0.06s trên CPU).
- Thuật toán Marching Cubes trích xuất bề mặt tại `level=0.0`, tự động tạo màng lưới đơn nhất (watertight), loại bỏ 100% ngấn phân chia và mặt cắt rác bên trong.

### 3.3. Anatomical Coloring (Vertex Colors)
- Dựa vào trường tọa độ không gian của các đỉnh sau khi trích xuất lưới:
  - Khu vực đầu & khuôn mặt ($Z > 1.48$): phân vùng da mặt, mắt, môi, mũi.
  - Khu vực tóc ($Z > 1.50$ phía sau và đỉnh đầu): màu tóc sẫm/đen/tùy biến.
  - Khu vực thân & tứ chi: màu trang phục/giáp/da theo archetype (`GHOST_ROBED`, `MECHA_HEAVY`, `WARRIOR_HUMANOID`).

### 3.4. DCC & Post-Processing
- Định hướng chuẩn Y-up cho glTF/Three.js.
- Căn chỉnh điểm chân tiếp sàn $Y=0$.
- Cập nhật pipeline kiểm định rig landmarks tương thích với hình dạng mới.

## 4. Verification & Test Plan
- Unit tests: `tests/test_procedural_generator.py` kiểm tra mesh sinh ra là đơn nhất (`len(mesh.faces) > 10000`, `mesh.is_watertight` hoặc manifold).
- Integration tests: `tests/test_gateway.py` sinh character hoàn chỉnh qua API và xác nhận định dạng `.glb`.
- UI Visual Verification: Xem trực tiếp mô hình trên trình duyệt qua WebGL 3D Studio.
