import os
from pathlib import Path
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    app_name: str = "Render3D AI Engine"
    blender_executable: str = os.getenv("BLENDER_PATH", "blender")
    default_poly_budget_character: int = 25000
    default_poly_budget_weapon: int = 8000
    pbr_texture_resolution: int = 2048

def get_gemini_api_key(key_file_path: str = "api-key.txt") -> str:
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    path = Path(key_file_path)
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content

    raise ValueError("Gemini API Key not found. Please set GEMINI_API_KEY environment variable or provide api-key.txt.")
