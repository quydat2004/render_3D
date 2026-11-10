import os
import pytest
from services.common.config import get_gemini_api_key, AppSettings

def test_get_gemini_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_env_key_12345")
    assert get_gemini_api_key() == "test_env_key_12345"

def test_get_gemini_api_key_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fake_key_file = tmp_path / "api-key.txt"
    fake_key_file.write_text("file_key_67890", encoding="utf-8")

    key = get_gemini_api_key(key_file_path=str(fake_key_file))
    assert key == "file_key_67890"

def test_get_gemini_api_key_missing_raises_error(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    non_existent = tmp_path / "non_existent.txt"
    with pytest.raises(ValueError, match="Gemini API Key not found"):
        get_gemini_api_key(key_file_path=str(non_existent))

def test_app_settings_defaults():
    settings = AppSettings()
    assert settings.app_name == "Render3D AI Engine"
    assert settings.blender_executable in ["blender", settings.blender_executable]
