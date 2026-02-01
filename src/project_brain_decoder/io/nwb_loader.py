from __future__ import annotations
from pathlib import Path
from pynwb import NWBHDF5IO
import numpy as np


def get_project_root() -> Path:
    # Anchor at this file, then go up to the repo root
    return Path(__file__).resolve().parents[3]


def load_nwb(file: str, root: Path | None = None) -> dict[str, np.array]:
    """
    Loading an NWB file from the project's raw data directory.

    :param file: str (Name of the NWB file)
    :param root: repo root of the project
    :return: a dict with features and target arrays
    """
    root = get_project_root()
    path = root / "data" / "raw" / file
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {path}")
    with NWBHDF5IO(path.as_posix(), 'r') as io:
        nwb_file = io.read()
        neural_spiking_band = nwb_file.analysis["SpikingBandPower"].data[:]
        neural_threshold_crossings = nwb_file.analysis["ThresholdCrossings"].data[:]
        target_index_position = nwb_file.analysis["index_position"].data[:]
        target_index_velocity = nwb_file.analysis["index_velocity"].data[:]
        target_mrs_position = nwb_file.analysis["mrs_position"].data[:]
        target_mrs_velocity = nwb_file.analysis["mrs_velocity"].data[:]

        array_dict = {"neural_spiking_band": neural_spiking_band,
                      "neural_threshold_crossings": neural_threshold_crossings,
                      "target_index_position": target_index_position,
                      "target_index_velocity": target_index_velocity,
                      "target_mrs_position": target_mrs_position,
                      "target_mrs_velocity": target_mrs_velocity}
        return array_dict