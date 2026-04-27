from __future__ import annotations
from pathlib import Path
from pynwb import NWBHDF5IO
import numpy as np


def load_nwb(file_path: Path) -> dict[str, np.ndarray]:
    """
    Load a single NWB file. For multiple sessions, the caller should loop over
    files and call load_nwb once per file.

    :param file_path: full path to one NWB file
    :return: dict with feature and target arrays
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {path}")
    with NWBHDF5IO(path.as_posix(), "r") as io:
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

