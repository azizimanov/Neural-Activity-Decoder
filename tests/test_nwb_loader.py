import numpy as np
import pytest
from project_brain_decoder.config import get_project_root
from project_brain_decoder.io.nwb_loader import load_nwb


@pytest.fixture
def sample_nwb():
    """Use the first committed session as a test fixture."""
    path = get_project_root() / "data" / "raw" / "sub-Monkey-N_ses-20200127_ecephys.nwb"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return load_nwb(path)


def test_required_keys(sample_nwb):
    expected = {
        "neural_spiking_band",
        "neural_threshold_crossings",
        "target_index_velocity",
        "target_mrs_velocity",
    }
    assert expected.issubset(sample_nwb.keys())


def test_channel_count(sample_nwb):
    assert sample_nwb["neural_spiking_band"].shape[1] == 96
    assert sample_nwb["neural_threshold_crossings"].shape[1] == 96


def test_time_alignment(sample_nwb):
    """Neural and target arrays must have matching time dimensions."""
    T_neural = sample_nwb["neural_spiking_band"].shape[0]
    assert sample_nwb["neural_threshold_crossings"].shape[0] == T_neural
    assert len(sample_nwb["target_index_velocity"]) == T_neural
    assert len(sample_nwb["target_mrs_velocity"]) == T_neural


def test_no_nans(sample_nwb):
    for key in ["neural_spiking_band", "neural_threshold_crossings",
                "target_index_velocity", "target_mrs_velocity"]:
        assert not np.isnan(sample_nwb[key]).any(), f"NaNs found in {key}"