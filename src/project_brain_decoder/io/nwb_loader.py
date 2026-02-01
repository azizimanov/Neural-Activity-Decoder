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
    :return: a dict with feature and target data
    """
    root = get_project_root()
    path = root / "data" / "raw" / file
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {path}")
    with NWBHDF5IO(path.as_posix(), 'r') as io:
        # spiking_band_power = io.analysis["SpikingBandPower"]
        nwb_file = io.read()
        spiking_band = nwb_file.analysis["SpikingBandPower"].data[:]
        threshold_crossings = nwb_file.analysis["ThresholdCrossings"].data[:]
        index_position = nwb_file.analysis["index_position"].data[:]
        index_velocity = nwb_file.analysis["index_velocity"].data[:]
        mrs_position = nwb_file.analysis["mrs_position"].data[:]
        mrs_velocity = nwb_file.analysis["mrs_velocity"].data[:]

        array_dict = {"spiking_band": spiking_band,
                      "threshold_crossings": threshold_crossings,
                      "index_position": index_position,
                      "index_velocity": index_velocity,
                      "mrs_position": mrs_position,
                      "mrs_velocity": mrs_velocity}
        return array_dict