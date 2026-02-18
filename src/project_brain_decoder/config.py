from pathlib import Path

def get_project_root() -> Path:
    # Anchor at this file, then go up to the repo root
    return Path(__file__).resolve().parents[2]