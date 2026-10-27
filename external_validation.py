"""External validation of recurrence-risk models: develop on Rotterdam, validate on GBSG2.

Development cohort: node-positive patients from the Rotterdam tumor bank
(``survival::rotterdam``, n = 1,546), follow-up censored at 7 years to match
GBSG2. Validation cohort: all 686 GBSG2 trial patients, used once, never for
fitting or tuning. Outcome in both: recurrence-free survival (recurrence or
death). Predictors are those recorded in both cohorts, harmonized: age,
menopausal status, tumor size category, grade 3 vs 1-2 (Rotterdam does not
separate grades 1 and 2), positive nodes, progesterone and estrogen receptors,
hormone therapy.

Models, from naive to flexible:

* ``null_km``: Kaplan-Meier of the development cohort (same risk for everyone);
* ``nodes_only``: Cox model on log(1 + nodes), the classic single prognostic factor;
* ``cox_linear``: Cox model, all predictors entered linearly;
* ``cox_log``: Cox model with prespecified log(1 + x) transforms for the three
  right-skewed counts/assays (nodes, PgR, ER);
* ``gbm``: gradient-boosted Cox model (scikit-survival), hyperparameters chosen
  by 5-fold cross-validation inside the development cohort.

Validation metrics at the 5-year horizon, each with a 95% bootstrap CI over the
validation cohort: Harrell's C, Uno's C, integrated Brier score (0.5 to 5
years), index of prediction accuracy (IPA, Brier skill vs the null model),
observed/expected 5-year risk and calibration slope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sksurv.ensemble import GradientBoostingSurvivalAnalysis  # noqa: E402
from sksurv.linear_model import CoxPHSurvivalAnalysis  # noqa: E402
from sksurv.metrics import brier_score, concordance_index_ipcw, integrated_brier_score  # noqa: E402
from sksurv.nonparametric import kaplan_meier_estimator  # noqa: E402
from sksurv.util import Surv  # noqa: E402
from statsmodels.duration.hazard_regression import PHReg  # noqa: E402

from analysis import harrell_c  # noqa: E402

ROOT = Path(__file__).resolve().parent
YEAR = 365.25
HORIZON = 5 * YEAR
DEV_CENSOR = 7 * YEAR
SEED = 20261023
N_BOOT = 500
MODELS = ["null_km", "nodes_only", "cox_linear", "cox_log", "gbm"]
LABELS = {"null_km": "Null (Kaplan-Meier)", "nodes_only": "Cox, nodes only", "cox_linear": "Cox, linear terms",
          "cox_log": "Cox, log-transformed markers", "gbm": "Gradient-boosted Cox"}
GBM_GRID = [{"learning_rate": lr, "n_estimators": n, "max_depth": d}
            for lr in (0.05, 0.1) for n in (300, 600, 1000) for d in (1, 2)]


# ------------------------------------------------------------------------ data
def load_development(path: str | Path | None = None) -> pd.DataFrame:
    """Rotterdam node-positive patients with RFS = first of recurrence or death, censored at 7 years."""
    r = pd.read_csv(Path(path) if path else ROOT / "data" / "rotterdam.csv")
    if len(r) != 2982:
        raise ValueError("unexpected Rotterdam row count")
    r = r[r.nodes > 0].copy()
    time = np.where(r.recur == 1, r.rtime, r.dtime).astype(float)
    event = np.maximum(r.recur, r.death).astype(int)
    over = time > DEV_CENSOR
    out = pd.DataFrame({"age": r.age, "meno": r.meno, "size": r["size"], "grade3": (r.grade == 3).astype(int),
                        "nodes": r.nodes, "pgr": r.pgr, "er": r.er, "hormon": r.hormon,
                        "time": np.where(over, DEV_CENSOR, time), "event": np.where(over, 0, event)})
    return out.reset_index(drop=True)


def load_validation(path: str | Path | None = None) -> pd.DataFrame:
    """GBSG2 harmonized to the development coding (tumor size in mm -> Rotterdam categories)."""
    g = pd.read_csv(Path(path) if path else ROOT / "data" / "gbsg2.csv")
    size = pd.cut(g.tsize, [0, 20, 50, np.inf], labels=["<=20", "20-50", ">50"])
    return pd.DataFrame({"age": g.age, "meno": (g.menostat == "Post").astype(int), "size": size.astype(str),
                         "grade3": (g.tgrade == "III").astype(int), "nodes": g.pnodes, "pgr": g.progrec,
                         "er": g.estrec, "hormon": (g.horTh == "yes").astype(int),
                         "time": g.time_days.astype(float), "event": g.event.astype(int)})


def features(df: pd.DataFrame, model: str) -> pd.DataFrame:
    if model == "nodes_only":
        return pd.DataFrame({"log_nodes": np.log1p(df.nodes)})
    X = pd.DataFrame({"age": df.age, "meno": df.meno, "size_20_50": (df["size"] == "20-50").astype(int),
                      "size_gt50": (df["size"] == ">50").astype(int), "grade3": df.grade3, "hormon": df.hormon})
    if model == "cox_linear":
        X[["nodes", "pgr", "er"]] = df[["nodes", "pgr", "er"]]
    else:  # cox_log and gbm share the transformed inputs (trees are invariant to monotone transforms anyway)
        X["log_nodes"], X["log_pgr"], X["log_er"] = np.log1p(df.nodes), np.log1p(df.pgr), np.log1p(df.er)
    return X.astype(float)


def surv(df: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(df.event.astype(bool), df.time)


# ---------------------------------------------------------------------- models
def tune_gbm(dev: pd.DataFrame, seed: int = SEED) -> tuple[dict, pd.DataFrame]:
    """5-fold CV inside the development cohort, scored by Uno's C at 5 years."""
    X, y = features(dev, "gbm"), surv(dev)
    rows = []
    for params in GBM_GRID:
        scores = []
        for train, test in KFold(5, shuffle=True, random_state=seed).split(X):
            m = GradientBoostingSurvivalAnalysis(loss="coxph", subsample=1.0, random_state=seed, **params)
            m.fit(X.iloc[train], y[train])
            scores.append(concordance_index_ipcw(y[train], y[test], m.predict(X.iloc[test]), tau=HORIZON)[0])
        rows.append({**params, "cv_uno_c": float(np.mean(scores)), "cv_uno_c_sd": float(np.std(scores, ddof=1))})
    table = pd.DataFrame(rows).sort_values("cv_uno_c", ascending=False).reset_index(drop=True)
    best = table.iloc[0][["learning_rate", "n_estimators", "max_depth"]].to_dict()
    best = {"learning_rate": float(best["learning_rate"]), "n_estimators": int(best["n_estimators"]),
            "max_depth": int(best["max_depth"])}
    return best, table


def fit_models(dev: pd.DataFrame, gbm_params: dict, seed: int = SEED) -> dict:
    y = surv(dev)
    fitted = {}
    for name in ["nodes_only", "cox_linear", "cox_log"]:
        fitted[name] = CoxPHSurvivalAnalysis(ties="breslow").fit(features(dev, name), y)
    fitted["gbm"] = GradientBoostingSurvivalAnalysis(loss="coxph", subsample=1.0, random_state=seed,
                                                     **gbm_params).fit(features(dev, "gbm"), y)
    return fitted


def predict(fitted: dict, dev: pd.DataFrame, val: pd.DataFrame, times: np.ndarray) -> dict[str, dict]:
    """Risk score (higher = worse) and survival curves at ``times`` for each model."""
    km_t, km_s = kaplan_meier_estimator(dev.event.astype(bool), dev.time)
    km_at = km_s[np.searchsorted(km_t, times, side="right") - 1]
    out = {"null_km": {"risk": np.zeros(len(val)), "surv": np.tile(km_at, (len(val), 1))}}
    for name, model in fitted.items():
        X = features(val, name)
        out[name] = {"risk": model.predict(X),
                     "surv": np.vstack([fn(times) for fn in model.predict_survival_function(X)])}
    return out


# ------------------------------------------------------------------ evaluation
def calibration_slope(val: pd.DataFrame, risk5: np.ndarray) -> float:
    """Cox regression of the validation outcome on cloglog(predicted 5-year risk); ideal = 1.
    For a Cox model this equals the classic slope on the linear predictor."""
    x = np.log(-np.log(np.clip(1 - risk5, 1e-12, 1 - 1e-12)))
    if np.allclose(x, x[0]):
        return float("nan")
    fit = PHReg(val.time, x[:, None], status=val.event, ties="breslow").fit(disp=0)
    return float(np.asarray(fit.params)[0])


def km_risk(val: pd.DataFrame, t: float = HORIZON) -> float:
    km_t, km_s = kaplan_meier_estimator(val.event.astype(bool), val.time)
    return float(1 - km_s[np.searchsorted(km_t, t, side="right") - 1])


def metrics(dev: pd.DataFrame, val: pd.DataFrame, pred: dict, times: np.ndarray) -> pd.DataFrame:
    y_dev, y_val = surv(dev), surv(val)
    i5 = int(np.argmin(np.abs(times - HORIZON)))
    ibs_null = integrated_brier_score(y_dev, y_val, pred["null_km"]["surv"], times)
    observed = km_risk(val)
    rows = []
    for name in MODELS:
        p = pred[name]
        risk5 = 1 - p["surv"][:, i5]
        ibs = integrated_brier_score(y_dev, y_val, p["surv"], times)
        tied = name == "null_km"
        rows.append({
            "model": name,
            "harrell_c": 0.5 if tied else harrell_c(val.time.to_numpy(), val.event.to_numpy(), p["risk"]),
            "uno_c_5y": 0.5 if tied else float(concordance_index_ipcw(y_dev, y_val, p["risk"], tau=HORIZON)[0]),
            "brier_5y": float(brier_score(y_dev, y_val, p["surv"][:, i5], times[i5])[1][0]),
            "ibs": float(ibs), "ipa": float(1 - ibs / ibs_null),
            "predicted_risk_5y": float(risk5.mean()), "observed_risk_5y": observed,
            "o_e_ratio": observed / float(risk5.mean()),
            "calibration_slope": float("nan") if tied else calibration_slope(val, risk5),
        })
    return pd.DataFrame(rows)


def bootstrap(dev: pd.DataFrame, val: pd.DataFrame, pred: dict, times: np.ndarray, n_boot: int = N_BOOT,
              seed: int = SEED) -> pd.DataFrame:
    """Resample the validation cohort; models stay fixed. Returns one row per (replicate, model)."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, len(val), len(val))
        vb = val.iloc[idx].reset_index(drop=True)
        if vb.time.max() <= times[-1]:
            continue
        pb = {m: {"risk": p["risk"][idx], "surv": p["surv"][idx]} for m, p in pred.items()}
        rows.append(metrics(dev, vb, pb, times).assign(replicate=b))
    return pd.concat(rows, ignore_index=True)


def summarize(point: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    cols = ["harrell_c", "uno_c_5y", "brier_5y", "ibs", "ipa", "o_e_ratio", "calibration_slope"]
    out = point.set_index("model").copy()
    for c in cols:
        q = boot.groupby("model")[c].quantile([0.025, 0.975]).unstack()
        out[f"{c}_ci_low"], out[f"{c}_ci_high"] = q[0.025], q[0.975]
    # Paired differences against the strongest simple model.
    ref = boot[boot.model == "cox_log"].set_index("replicate")
    for m in MODELS:
        d = boot[boot.model == m].set_index("replicate")
        for c in ["uno_c_5y", "ipa"]:
            diff = d[c] - ref[c]
            out.loc[m, f"{c}_minus_cox_log_ci_low"] = diff.quantile(0.025)
            out.loc[m, f"{c}_minus_cox_log_ci_high"] = diff.quantile(0.975)
    return out.reset_index()


def calibration_table(val: pd.DataFrame, risk5: np.ndarray, groups: int = 5) -> pd.DataFrame:
    """Observed (Kaplan-Meier) vs mean predicted 5-year risk by quintile of predicted risk."""
    q = pd.qcut(risk5, groups, labels=False, duplicates="drop")
    rows = []
    for k in sorted(np.unique(q)):
        m = q == k
        rows.append({"group": int(k) + 1, "n": int(m.sum()), "predicted": float(risk5[m].mean()),
                     "observed": km_risk(val[m])})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, calib: dict[str, pd.DataFrame], path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    s = summary.set_index("model").loc[MODELS[::-1]]
    y = np.arange(len(s))
    a.errorbar(s.uno_c_5y, y, xerr=[s.uno_c_5y - s.uno_c_5y_ci_low, s.uno_c_5y_ci_high - s.uno_c_5y],
               fmt="o", color="#1f4e79", capsize=3)
    a.axvline(0.5, color="grey", ls="--", lw=0.8)
    a.set_yticks(y, [LABELS[m] for m in s.index])
    a.set(xlabel="Uno's C at 5 years (95% bootstrap CI)", title="Discrimination in GBSG2 (external)")
    a.grid(axis="x", alpha=0.3)
    for name, color in [("cox_log", "#1f4e79"), ("gbm", "#e67e22"), ("nodes_only", "#7f8c8d")]:
        c = calib[name]
        b.plot(c.predicted, c.observed, marker="o", color=color, label=LABELS[name])
    lim = [0, 0.9]
    b.plot(lim, lim, color="black", ls="--", lw=0.8, label="perfect calibration")
    b.set(xlim=lim, ylim=lim, xlabel="Predicted 5-year recurrence risk (Rotterdam model)",
          ylabel="Observed 5-year risk in GBSG2 (Kaplan-Meier)", title="Calibration by quintile of predicted risk")
    b.legend(frameon=False, fontsize=8); b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------- run
def run(output_dir: str | Path | None = None, figure_dir: str | Path | None = None, n_boot: int = N_BOOT) -> dict:
    out = Path(output_dir) if output_dir else ROOT / "outputs"
    figs = Path(figure_dir) if figure_dir else (ROOT / "figures" if output_dir is None else out)
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    dev, val = load_development(), load_validation()
    gbm_params, cv = tune_gbm(dev)
    cv.to_csv(out / "external_gbm_cv.csv", index=False)
    fitted = fit_models(dev, gbm_params)
    coefs = pd.concat([pd.DataFrame({"model": n, "term": features(dev, n).columns, "log_hr": fitted[n].coef_})
                       for n in ["nodes_only", "cox_linear", "cox_log"]], ignore_index=True)
    coefs.to_csv(out / "external_cox_coefficients.csv", index=False)

    times = np.linspace(0.5 * YEAR, HORIZON, 46)
    pred = predict(fitted, dev, val, times)
    point = metrics(dev, val, pred, times)
    boot = bootstrap(dev, val, pred, times, n_boot)
    summary = summarize(point, boot)
    summary.to_csv(out / "external_validation.csv", index=False)
    i5 = int(np.argmin(np.abs(times - HORIZON)))
    calib = {m: calibration_table(val, 1 - pred[m]["surv"][:, i5]) for m in ["nodes_only", "cox_log", "gbm"]}
    pd.concat([c.assign(model=m) for m, c in calib.items()]).to_csv(out / "external_calibration.csv", index=False)
    plot(summary, calib, figs / "external_validation.png")

    s = summary.set_index("model")
    result = {
        "development": {"cohort": "Rotterdam, node-positive", "n": int(len(dev)), "events": int(dev.event.sum())},
        "validation": {"cohort": "GBSG2", "n": int(len(val)), "events": int(val.event.sum()),
                       "observed_risk_5y": float(s.loc["null_km", "observed_risk_5y"])},
        "horizon_years": 5, "n_boot": int(boot.replicate.nunique()), "gbm_params": gbm_params,
        "models": {m: {k: (None if pd.isna(v) else float(v)) for k, v in s.loc[m].items()} for m in MODELS},
    }
    (out / "external_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir"); p.add_argument("--figure-dir"); p.add_argument("--n-boot", type=int, default=N_BOOT)
    a = p.parse_args()
    r = run(a.output_dir, a.figure_dir, a.n_boot)
    cols = ["harrell_c", "uno_c_5y", "ibs", "ipa", "o_e_ratio", "calibration_slope"]
    print(pd.DataFrame(r["models"]).T[cols].round(3).to_string())
