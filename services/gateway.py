import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import trimesh

from services.common.config import AppSettings
from services.orchestrator.gemini_orchestrator import (
    GeminiOrchestrator,
    AssetType,
    OrthographicConceptPrompt,
    LevelLayoutSchema,
    QAInspectionResult
)
from services.dcc.blender_pipeline_processor import BlenderPipelineProcessor
from services.dcc.character_rigging_engine import CharacterRiggingEngine
from services.dcc.weapon_hard_surface import WeaponHardSurfaceEngine, WeaponSocket
from services.dcc.map_modular_assembler import ModularMapAssembler

settings = AppSettings()
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="End-to-End AI Engine for 3D Characters, Weapons, and Environments powered by Gemini & DCC Pipelines"
)

ASSETS_DIR = Path("storage/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

orchestrator = GeminiOrchestrator()
blender_proc = BlenderPipelineProcessor(blender_path=settings.blender_executable)
char_rigger = CharacterRiggingEngine()
weapon_engine = WeaponHardSurfaceEngine()
map_assembler = ModularMapAssembler()

class CharacterRequest(BaseModel):
    prompt: str = Field(description="Character description")
    style: Optional[str] = Field(default="Game-Ready Realistic")
    poly_budget: Optional[int] = Field(default=25000)

class WeaponRequest(BaseModel):
    prompt: str = Field(description="Weapon description")
    weapon_type: Optional[str] = Field(default="ranged")
    poly_budget: Optional[int] = Field(default=8000)

class MapRequest(BaseModel):
    prompt: str = Field(description="Level/Environment description")
    dimensions: Optional[List[float]] = Field(default=[40.0, 40.0])

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "blender_available": blender_proc.blender_available
    }

@app.get("/api/v1/assets/{filename}")
def get_asset(filename: str):
    file_path = ASSETS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path=file_path, media_type="model/gltf-binary", filename=filename)

@app.post("/api/v1/generate/character")
def generate_character(req: CharacterRequest):
    task_id = str(uuid.uuid4())[:8]

    # 1. Gemini Cognitive Concept Expansion
    concept = orchestrator.expand_asset_prompt(req.prompt, AssetType.CHARACTER)

    # 2. Generative 3D Mesh (base humanoid geometry)
    raw_mesh = trimesh.creation.cylinder(radius=0.25, height=1.8)
    raw_mesh.apply_translation([0, 0, 0.9])

    raw_obj_path = ASSETS_DIR / f"char_raw_{task_id}.obj"
    raw_mesh.export(str(raw_obj_path))

    # 3. DCC Retopology & UV Processing
    clean_glb_path = ASSETS_DIR / f"char_clean_{task_id}.glb"
    tt_dir = ASSETS_DIR / f"turntable_{task_id}"

    proc_result = blender_proc.process_mesh(
        input_path=str(raw_obj_path),
        output_path=str(clean_glb_path),
        target_faces=req.poly_budget or settings.default_poly_budget_character,
        turntable_dir=str(tt_dir)
    )

    # 4. Character Rigging & Skinning
    landmarks = char_rigger.estimate_landmarks(raw_mesh)

    # 5. Gemini VLM Turntable QA Pass
    image_bytes = []
    for frame_path in proc_result.get("turntable_frames", []):
        if Path(frame_path).exists():
            image_bytes.append(Path(frame_path).read_bytes())

    qa_report = orchestrator.inspect_turntable(image_bytes, AssetType.CHARACTER)

    return {
        "task_id": task_id,
        "status": "completed",
        "asset_type": "character",
        "concept": concept.model_dump(),
        "landmarks": landmarks.model_dump(),
        "model_url": f"/api/v1/assets/char_clean_{task_id}.glb",
        "qa_report": qa_report.model_dump()
    }

@app.post("/api/v1/generate/weapon")
def generate_weapon(req: WeaponRequest):
    task_id = str(uuid.uuid4())[:8]

    # 1. Gemini Cognitive Expansion
    concept = orchestrator.expand_asset_prompt(req.prompt, AssetType.WEAPON)

    # 2. Hard-Surface Base Mesh
    raw_weapon = trimesh.creation.box(extents=[0.08, 0.85, 0.25])
    raw_weapon.apply_translation([0, 0.2, 0])

    raw_obj_path = ASSETS_DIR / f"weapon_raw_{task_id}.obj"
    raw_weapon.export(str(raw_obj_path))

    # 3. Retopology & Decimation
    clean_glb_path = ASSETS_DIR / f"weapon_{task_id}.glb"
    tt_dir = ASSETS_DIR / f"weapon_tt_{task_id}"

    proc_result = blender_proc.process_mesh(
        input_path=str(raw_obj_path),
        output_path=str(clean_glb_path),
        target_faces=req.poly_budget or settings.default_poly_budget_weapon,
        turntable_dir=str(tt_dir)
    )

    # 4. Detect Attachment Sockets & UCX Collision
    sockets = weapon_engine.detect_sockets(raw_weapon)
    collision_hull = weapon_engine.generate_collision_hull(raw_weapon)

    # 5. Gemini VLM QA Pass
    image_bytes = []
    for frame in proc_result.get("turntable_frames", []):
        if Path(frame).exists():
            image_bytes.append(Path(frame).read_bytes())

    qa_report = orchestrator.inspect_turntable(image_bytes, AssetType.WEAPON)

    return {
        "task_id": task_id,
        "status": "completed",
        "asset_type": "weapon",
        "concept": concept.model_dump(),
        "sockets": [s.model_dump() for s in sockets],
        "collision_hull_faces": len(collision_hull.faces),
        "model_url": f"/api/v1/assets/weapon_{task_id}.glb",
        "qa_report": qa_report.model_dump()
    }

@app.post("/api/v1/generate/map")
def generate_map(req: MapRequest):
    task_id = str(uuid.uuid4())[:8]

    # 1. Gemini Spatial Level Planning
    layout = orchestrator.plan_environment(req.prompt)
    if req.dimensions:
        layout.dimensions = req.dimensions

    # 2. Modular Assembly & Boundary Snapping
    assembly = map_assembler.assemble_scene(layout, output_dir=str(ASSETS_DIR))

    return {
        "task_id": task_id,
        "status": "completed",
        "asset_type": "environment",
        "level_id": layout.level_id,
        "dimensions": layout.dimensions,
        "grid_size_m": layout.grid_size_m,
        "model_url": f"/api/v1/assets/{layout.level_id}.glb",
        "navmesh": assembly.get("navmesh", {}),
        "classified_props": assembly.get("classified_props", [])
    }
