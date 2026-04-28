import pandas as pd
import pytest
from project_brain_decoder.eval import save_r2


@pytest.fixture
def tmp_results_dir(tmp_path, monkeypatch):
    """Redirect get_project_root() to a tmp dir so tests don't touch real results/."""
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(
        "project_brain_decoder.eval.save_r2.get_project_root",
        lambda: tmp_path,
    )
    return results


def test_creates_csv(tmp_results_dir):
    save_r2.get_scores(model="test_model", score1=0.5, score2=0.6)
    csv_path = tmp_results_dir / "r2_scores.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["Model", "Index vel. R2 Score", "MRS vel. R2 Score"]


def test_appends_new_model(tmp_results_dir):
    save_r2.get_scores(model="model_a", score1=0.1, score2=0.2)
    save_r2.get_scores(model="model_b", score1=0.3, score2=0.4)
    df = pd.read_csv(tmp_results_dir / "r2_scores.csv")
    assert len(df) == 2
    assert set(df["Model"]) == {"model_a", "model_b"}


def test_updates_existing_model(tmp_results_dir):
    save_r2.get_scores(model="model_a", score1=0.1, score2=0.2)
    save_r2.get_scores(model="model_a", score1=0.5, score2=0.6)
    df = pd.read_csv(tmp_results_dir / "r2_scores.csv")
    assert len(df) == 1
    assert df.iloc[0]["Index vel. R2 Score"] == 0.5
    assert df.iloc[0]["MRS vel. R2 Score"] == 0.6