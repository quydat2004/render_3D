import os
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

    # Hunyuan3D returns tuple with dict containing file path
    mock_job.result.return_value = ({"value": str(fake_glb)}, "html_out", {}, 1234)
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
