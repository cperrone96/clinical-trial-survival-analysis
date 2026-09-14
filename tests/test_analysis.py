import json
from pathlib import Path

import pandas as pd

import analysis


ROOT = Path(__file__).parents[1]


def test_input_contract_and_counts():
    df = analysis.load_data()
    assert len(df) == 686
    assert int(df.event.sum()) == 299
    assert (df.event == 0).sum() == 387
    assert set(df.tgrade) == {"I", "II", "III"}


def test_grade_contract_is_shared_and_revision_removes_grade_term():
    df = analysis.load_data()
    assert analysis.design_matrix(df).grade_ord.tolist()[:4] == [2, 2, 2, 2]
    assert "grade_ord" not in analysis.design_matrix(df, include_grade=False)


def test_deterministic_metrics_and_outputs(tmp_path):
    metrics = analysis.run(output_dir=tmp_path)
    assert metrics["n"] == 686
    assert metrics["events"] == 299
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "ph_diagnostics.csv").exists()
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert loaded["grade_coding"] == {"I": 1, "II": 2, "III": 3}
    assert not any("logtime" in term for term in loaded["revised_terms"])
    assert "ph_limitation" in loaded
    flagged = {x["term"] for x in loaded["diagnostics"]["proportional_hazards"] if x["p_value"] < 0.05}
    assert {"age", "menostat_Post", "grade_ord", "estrec"}.issubset(flagged)
