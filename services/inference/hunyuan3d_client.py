import os
import time
import shutil
import logging
from pathlib import Path
from typing import Optional, Any

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None
    handle_file = None

logger = logging.getLogger(__name__)

class Hunyuan3DClient:
    SPACE_ID = "tencent/Hunyuan3D-2"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("HF_TOKEN="):
                        self.token = line.split("=", 1)[1].strip()
                        break

        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not Client:
            raise RuntimeError("gradio_client is not installed.")
        self._client = Client(self.SPACE_ID, token=self.token)
        return self._client

    def generate_3d_from_image(
        self,
        image_path: str,
        output_path: str,
        caption: str = "3D character model, detailed geometry, game-ready",
        steps: int = 15,
        timeout: int = 180
    ) -> str:
        client = self._get_client()
        img_arg = handle_file(image_path) if handle_file else image_path

        job = client.submit(
            caption=caption,
            image=img_arg,
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=steps,
            guidance_scale=4.5,
            seed=42,
            octree_resolution=256,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=True,
            api_name="/shape_generation"
        )

        start = time.time()
        while not job.done():
            if time.time() - start > timeout:
                raise TimeoutError(f"Hunyuan3D generation exceeded {timeout}s.")
            time.sleep(3)

        result = job.result()
        file_item = result[0]

        # Gradio return can be a dict {'value': path} or direct str path
        src_file = file_item["value"] if isinstance(file_item, dict) else str(file_item)

        out_dest = Path(output_path)
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_file, str(out_dest))
        return str(out_dest)

    def generate_3d_from_text(
        self,
        caption: str,
        output_path: str,
        steps: int = 20,
        timeout: int = 180
    ) -> str:
        client = self._get_client()
        job = client.submit(
            caption=caption,
            image=None,
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=steps,
            guidance_scale=5.0,
            seed=42,
            octree_resolution=256,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=True,
            api_name="/shape_generation"
        )

        start = time.time()
        while not job.done():
            if time.time() - start > timeout:
                raise TimeoutError(f"Hunyuan3D generation exceeded {timeout}s.")
            time.sleep(3)

        result = job.result()
        file_item = result[0]
        src_file = file_item["value"] if isinstance(file_item, dict) else str(file_item)

        out_dest = Path(output_path)
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_file, str(out_dest))
        return str(out_dest)
