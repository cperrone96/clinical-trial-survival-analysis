import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis

ROOT = Path(__file__).parents[1]
R_OUT = ROOT / "outputs-r"


@pytest.fixture(scope="module")
def df():
    return analysis.load_data()


@pytest.fixture(scope="module")
def fits(df):
    return analysis.fit_ph(df, stratified=False), analysis.fit_ph(df, stratified=True)


def test_input_contract_and_counts(df):
    assert len(df) == 686
    assert int(df.event.sum()) == 299
    assert int((df.event == 0).sum()) == 387
    assert set(df.tgrade) == {"I", "II", "III"}


def test_grade_is_ordinal_in_original_and_absent_in_primary(df):
    assert analysis.design_matrix(df).grade_ord.tolist()[:4] == [2, 2, 2, 2]
    assert "grade_ord" not in analysis.design_matrix(df, include_grade=False)


def test_harrell_c_invariants():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.ones(5)
    assert analysis.harrell_c(t, e, -t) == 1.0
    assert analysis.harrell_c(t, e, t) == 0.0
    assert analysis.harrell_c(t, e, np.zeros(5)) == 0.5
    # A censored shortest time cannot anchor a comparable pair.
    e2 = np.array([0, 1, 1, 1, 1])
    assert analysis.harrell_c(t, e2, np.array([-10.0, -2, -3, -4, -5])) == 1.0


def test_counting_process_split_reproduces_primary_fit(df, fits):
    """Guards the statsmodels entry-time convention used by the time-varying model."""
    long = analysis.split_at_event_times(df)
    X = analysis.design_matrix(df, include_grade=False).iloc[long.row.to_numpy()].reset_index(drop=True)
    split_fit = analysis.fit_counting_process(df, long, X)
    primary, _ = fits[1]
    np.testing.assert_allclose(split_fit.params, primary.params, rtol=1e-7, atol=1e-10)
    assert split_fit.llf == pytest.approx(primary.llf, abs=1e-8)


def test_bootstrap_concordance_is_seeded(df):
    a = analysis.optimism_corrected_c(df, n_boot=5, seed=1)
    b = analysis.optimism_corrected_c(df, n_boot=5, seed=1)
    assert a == b
    assert a["corrected"] < a["apparent"]


@pytest.mark.skipif(not R_OUT.exists(), reason="R outputs not generated")
class TestParityWithR:
    """Python must reproduce survival::coxph, cox.zph, concordance, survdiff and survfit."""

    def test_cox_coefficients(self, fits):
        for (fit, _), name in zip(fits, ["original", "primary"]):
            r = pd.read_csv(R_OUT / f"cox_{name}_coefficients.csv")
            np.testing.assert_allclose(np.asarray(fit.params), r.log_hr, rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(np.asarray(fit.bse), r.se, rtol=1e-6)

    def test_ph_score_test_matches_cox_zph(self, df, fits):
        (orig, X0), (prim, X1) = fits
        r = pd.read_csv(R_OUT / "ph_diagnostics.csv")
        py = pd.concat([analysis.ph_score_test(df, orig, X0),
                        analysis.ph_score_test(df, prim, X1, df.tgrade.to_numpy())], ignore_index=True)
        np.testing.assert_allclose(py.chisq, r.chisq, rtol=1e-6)

    def test_concordance(self, df, fits):
        (orig, _), (prim, _) = fits
        r = pd.read_csv(R_OUT / "metrics.csv").iloc[0]
        assert analysis.model_c(df, np.asarray(orig.params), False) == pytest.approx(r.original_concordance, abs=1e-9)
        assert analysis.model_c(df, np.asarray(prim.params), True) == pytest.approx(r.primary_concordance, abs=1e-9)

    def test_logrank_and_km(self, df):
        r = pd.read_csv(R_OUT / "logrank_tests.csv")
        for g, chisq in zip(r.group, r.chisq):
            assert analysis.logrank(df, g)["chisq"] == pytest.approx(chisq, rel=1e-8)
        rk = pd.read_csv(R_OUT / "km_summary.csv")
        km = pd.concat([analysis.km_summary(df, None), analysis.km_summary(df, "horTh"),
                        analysis.km_summary(df, "tgrade")], ignore_index=True)
        cols = ["median_days", "median_ci_low", "median_ci_high", "rfs_5y", "rfs_5y_ci_low", "rfs_5y_ci_high"]
        np.testing.assert_allclose(km[cols].astype(float).to_numpy(), rk[cols].to_numpy(dtype=float),
                                   rtol=1e-8, equal_nan=True)

    def test_time_varying_model_matches_tt(self, df):
        r = pd.read_csv(R_OUT / "time_varying_sensitivity.csv")
        py = analysis.time_varying_fit(df)
        np.testing.assert_allclose(py.log_hr, r.log_hr, rtol=1e-6, atol=1e-9)


def test_run_writes_consistent_outputs(tmp_path):
    metrics = analysis.run(output_dir=tmp_path, n_boot=10)
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert loaded["n"] == 686 and loaded["events"] == 299
    assert "grade_ord" not in loaded["primary_terms"]
    # Grade is the term the stratification removes; age, menopause and ER still fail PH.
    assert set(loaded["ph_flagged_primary"]) == {"age", "menostat_Post", "estrec"}
    sens = pd.read_csv(tmp_path / "hormone_therapy_sensitivity.csv")
    assert (sens.ci_high < 1).all()
    for name in ["km_summary.csv", "logrank_tests.csv", "cox_coefficients.csv", "ph_diagnostics.csv",
                 "functional_form_diagnostics.csv", "influence_diagnostics.csv", "time_varying_sensitivity.csv"]:
        assert (tmp_path / name).exists()
    assert metrics["concordance"]["primary"]["n_boot"] == 10
