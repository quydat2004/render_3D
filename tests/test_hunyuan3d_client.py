import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from services.inference.hunyuan3d_client import Hunyuan3DClient

def test_hunyuan3d_client_init_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
    client = Hunyuan3DClient()
    assert client.token == "hf_test_token_123"

def test_hunyuan3d_client_generates_mesh_mocked(tmp_path):
    mock_gradio = MagicMock()
    mock_job = MagicMock()
    mock_job.done.return_value = True

    fake_glb = tmp_path / "mock_model.glb"
    fake_glb.write_bytes(b"glTF mock binary data")

    # /generation_all returns (white mesh, textured mesh, html, stats, seed)
    mock_job.result.return_value = ({"value": str(fake_glb)}, {"value": str(fake_glb)}, "html_out", {}, 1234)
    mock_gradio.submit.return_value = mock_job

    client = Hunyuan3DClient(token="dummy_token")
    client._client = mock_gradio

    input_img = tmp_path / "fake_input.jpg"
    input_img.write_bytes(b"mock_image_bytes")

    out_file = tmp_path / "output.glb"
    result_path = client.generate_3d_from_image(
        image_path=str(input_img),
        output_path=str(out_file),
        caption="Sci-fi warrior"
    )

    assert result_path == str(out_file)
    assert out_file.exists()
    assert out_file.read_bytes() == b"glTF mock binary data"

def test_texture_comes_from_generation_all_not_dead_paint_space(tmp_path):
    """The standalone Paint space answers its SSE stream with an HTML page, so texture has to come
    from the main space's /generation_all, which returns the textured mesh directly."""
    mock_gradio = MagicMock()
    mock_job = MagicMock()
    mock_job.done.return_value = True
    fake_glb = tmp_path / "textured.glb"
    fake_glb.write_bytes(b"textured glb")
    mock_job.result.return_value = ({"value": str(fake_glb)}, {"value": str(fake_glb)}, "html", {}, 7)
    mock_gradio.submit.return_value = mock_job

    client = Hunyuan3DClient(token="dummy_token")
    client._client = mock_gradio
    client._paint_client = MagicMock()

    img = tmp_path / "in.png"
    img.write_bytes(b"img")
    client.generate_3d_from_image(str(img), str(tmp_path / "out.glb"))

    called = mock_gradio.submit.call_args.kwargs["api_name"]
    assert called == "/generation_all"
    client._paint_client.submit.assert_not_called()
    assert client.last_texture_source == "hunyuan3d-paint"


def test_generation_all_fallback_to_shape_generation_and_atlas(tmp_path):
    mock_gradio = MagicMock()
    fake_glb = tmp_path / "shape.glb"
    import trimesh
    box = trimesh.creation.box(extents=(0.6, 1.75, 0.4))
    box.export(str(fake_glb))

    # /generation_all fails with AppError
    def mock_submit(*args, **kwargs):
        api = kwargs.get("api_name")
        job = MagicMock()
        job.done.return_value = True
        if api == "/generation_all":
            raise RuntimeError("texgen_worker is not defined")
        elif api == "/shape_generation":
            job.result.return_value = [{"value": str(fake_glb)}, "html", {}, 1234]
            return job
        return job

    mock_gradio.submit.side_effect = mock_submit

    client = Hunyuan3DClient(token="dummy_token")
    client._client = mock_gradio

    from PIL import Image
    ref_img = tmp_path / "ref.png"
    Image.new("RGBA", (128, 128), (220, 30, 30, 255)).save(str(ref_img))

    out_file = tmp_path / "out_atlas.glb"
    res = client.generate_3d_from_image(str(ref_img), str(out_file))

    assert Path(res).exists()
    scene = trimesh.load(res)
    geom = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    assert geom.visual.uv is not None
    assert geom.visual.material is not None


def test_generate_3d_untextured_clay_mode(tmp_path):
    """
    Mode texture=False: Generates pure 3D geometry sculpt (clay model) directly matching
    reference silhouette without texture maps or color bleeding.
    """
    mock_gradio = MagicMock()
    fake_glb = tmp_path / "raw_shape.glb"
    import trimesh
    box = trimesh.creation.box(extents=(0.6, 1.75, 0.4))
    box.export(str(fake_glb))

    job = MagicMock()
    job.done.return_value = True
    job.result.return_value = [{"value": str(fake_glb)}]
    mock_gradio.submit.return_value = job

    client = Hunyuan3DClient(token="dummy_token")
    client._client = mock_gradio

    ref_img = tmp_path / "ref_char.png"
    from PIL import Image
    Image.new("RGBA", (64, 64), (200, 50, 50, 255)).save(str(ref_img))

    out_file = tmp_path / "out_clay.glb"
    res = client.generate_3d_from_image(str(ref_img), str(out_file), texture=False)

    assert Path(res).exists()
    assert mock_gradio.submit.call_args.kwargs["api_name"] == "/shape_generation"
    assert client.last_texture_source is None

    scene = trimesh.load(res)
    geom = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene
    assert not hasattr(geom.visual, 'uv') or geom.visual.uv is None

