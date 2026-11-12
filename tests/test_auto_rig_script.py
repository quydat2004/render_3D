import sys
import pytest
from pathlib import Path

def test_script_exists():
    script_path = Path("scripts/auto_rig_and_animate.py")
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")
    assert "ARMATURE_AUTO" in content
    assert "export_animations" in content

def test_parse_cli_args():
    from scripts.auto_rig_and_animate import parse_cli_args
    args = ["blender", "--background", "--python", "script.py", "--", "input.glb", "output.glb", "walk,idle,run"]
    parsed = parse_cli_args(args)
    assert parsed["input_glb"] == "input.glb"
    assert parsed["output_glb"] == "output.glb"
    assert "walk" in parsed["animations"]
    assert "idle" in parsed["animations"]
    assert "run" in parsed["animations"]
