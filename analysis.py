"""Reproducible GBSG2 analysis (Python/statsmodels).

The estimand is recurrence-free survival (days to recurrence/death or censoring).
Tumor grade is coded ordinally (I=1, II=2, III=3) in the original model in all
three language implementations. Because its Schoenfeld residual trend is
diagnostic evidence against a constant grade HR, the primary revision
stratifies the Cox baseline hazard by grade and does not report a grade HR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.duration.hazard_regression import PHReg

ROOT = Path(__file__).resolve().parent
REQUIRED = ["horTh", "age", "menostat", "tsize", "tgrade", "pnodes", "progrec", "estrec", "time_days", "event"]
GRADE = {"I": 1, "II": 2, "III": 3}


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else ROOT / "data" / "gbsg2.csv"
    df = pd.read_csv(p)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if len(df) != 686 or not set(df.event.unique()).issubset({0, 1}):
        raise ValueError("unexpected GBSG2 schema/counts")
    return df


def design_matrix(df: pd.DataFrame, include_grade: bool = True) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["horTh_yes"] = (df.horTh == "yes").astype(float)
    out["age"] = df.age.astype(float)
    out["menostat_Post"] = (df.menostat.str.lower() == "post").astype(float)
    out["tsize"] = df.tsize.astype(float)
    if include_grade:
        out["grade_ord"] = df.tgrade.map(GRADE).astype(float)
    out["pnodes"] = df.pnodes.astype(float)
    out["progrec"] = df.progrec.astype(float)
    out["estrec"] = df.estrec.astype(float)
    return out


def fit_ph(df: pd.DataFrame, stratified: bool = False):
    X = design_matrix(df, include_grade=not stratified)
    # PHReg's strata argument is the grade contract used by the R/SAS scripts.
    model = PHReg(df.time_days, X, status=df.event, strata=df.tgrade if stratified else None)
    return model.fit(disp=0), X


def _coef_table(result, model_name: str) -> pd.DataFrame:
    names = list(result.model.exog_names)
    b = np.asarray(result.params)
    se = np.asarray(result.bse)
    return pd.DataFrame({"model": model_name, "term": names, "log_hr": b,
                         "hr": np.exp(b), "ci_low": np.exp(b - 1.96 * se),
                         "ci_high": np.exp(b + 1.96 * se), "p_value": result.pvalues})


def diagnostics(df: pd.DataFrame, result, X: pd.DataFrame) -> dict:
    """Return functional-form, influence and PH diagnostics without claims of validation."""
    mart = np.asarray(result.martingale_residuals)
    sch = np.asarray(result.schoenfeld_residuals)
    # statsmodels returns NaN Schoenfeld residuals for censored observations;
    # restrict to observed events explicitly.
    event = df.event.to_numpy(dtype=bool)
    log_time = np.log(df.time_days.to_numpy())
    terms = list(X.columns)
    ph = []
    for j, term in enumerate(terms):
        r = sch[:, j][event]
        rho, p = pearsonr(r, log_time[event]) if len(r) > 3 and np.std(r) else (np.nan, np.nan)
        ph.append({"term": term, "schoenfeld_logtime_r": float(rho), "p_value": float(p)})
    # Martingale residual correlation with each continuous covariate is a simple
    # functional-form screen; it is not a substitute for spline specification.
    functional = []
    for term in ["age", "tsize", "pnodes", "progrec", "estrec"]:
        keep = np.isfinite(df[term].to_numpy()) & np.isfinite(mart)
        rho, p = pearsonr(df.loc[keep, term], mart[keep])
        functional.append({"term": term, "residual_covariate_r": float(rho), "p_value": float(p)})
    score = np.asarray(result.score_residuals)
    influence = pd.DataFrame(score, columns=terms).replace([np.inf, -np.inf], np.nan).abs().max(axis=1).sort_values(ascending=False).head(10)
    return {"proportional_hazards": ph, "functional_form_screen": functional,
            "influence_screen": [{"row": int(i), "max_abs_score": float(v)} for i, v in influence.items()],
            "max_abs_martingale_residual": float(np.nanmax(np.abs(mart)))}


def run(data_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    out = Path(output_dir) if output_dir else ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    df = load_data(data_path)
    original, X0 = fit_ph(df, stratified=False)
    revised, X1 = fit_ph(df, stratified=True)
    original_table = _coef_table(original, "original_ordinal_grade")
    revised_table = _coef_table(revised, "revised_grade_stratified")
    coef = pd.concat([original_table, revised_table], ignore_index=True)
    coef.to_csv(out / "cox_coefficients.csv", index=False)
    diag = diagnostics(df, original, X0)
    pd.DataFrame(diag["proportional_hazards"]).to_csv(out / "ph_diagnostics.csv", index=False)
    pd.DataFrame(diag["functional_form_screen"]).to_csv(out / "functional_form_diagnostics.csv", index=False)
    pd.DataFrame(diag["influence_screen"]).to_csv(out / "influence_diagnostics.csv", index=False)
    # A compact, cross-language-ready input contract and parity counts.
    contract = df.copy(); contract["grade_ord"] = contract.tgrade.map(GRADE)
    contract.to_csv(out / "analysis_input_contract.csv", index=False)
    metrics = {"n": int(len(df)), "events": int(df.event.sum()), "censored": int((df.event == 0).sum()),
               "grade_counts": df.tgrade.value_counts().sort_index().to_dict(),
               "hormone_counts": df.horTh.value_counts().sort_index().to_dict(),
               "estimand": "recurrence-free survival in days; event=recurrence/death; censor=0",
               "grade_coding": GRADE, "original_terms": list(X0.columns), "revised_terms": list(X1.columns),
               "ph_limitation": "age, menopausal status, and estrogen receptor also show residual time trends; no outcome-leaking fixed time interactions are fitted, so their non-PH behavior remains a limitation",
               "original_loglik": float(original.llf), "revised_loglik": float(revised.llf),
               "diagnostics": diag}
    with open(out / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    # Minimal diagnostic plot: martingale residuals against fitted linear predictor.
    plt.figure(figsize=(6, 3.5)); plt.scatter(original.predict(X0).predicted_values, original.martingale_residuals, s=8, alpha=.45)
    plt.axhline(0, color="k", lw=.7); plt.xlabel("Cox linear predictor"); plt.ylabel("Martingale residual")
    plt.tight_layout(); plt.savefig(out / "figures" / "martingale_vs_lp.png", dpi=130); plt.close()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data"); parser.add_argument("--output-dir")
    args = parser.parse_args(); m = run(args.data, args.output_dir)
    print(json.dumps({k: m[k] for k in ("n", "events", "censored", "original_loglik", "revised_loglik")}, indent=2))
