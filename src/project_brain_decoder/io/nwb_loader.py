from __future__ import annotations
from pathlib import Path
from pynwb import NWBHDF5IO
from pynwb.file import NWBFile


def get_project_root() -> Path:
    # Anchor at this file, then go up to the repo root
    return Path(__file__).resolve().parents[3]


def load_nwb(file: str, root: Path | None = None) -> NWBFile:
    """
    Loading an NWB file from the project's raw data directory.

    :param file: str (Name of the NWB file)
    :param root: repo root of the project
    :return: NWBFile (The loaded NWB file as a Python project)
    """
    root = get_project_root()
    path = root / "data" / "raw" / file
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {path}")
    with NWBHDF5IO(path.as_posix(), 'r') as io:
        return io.read()