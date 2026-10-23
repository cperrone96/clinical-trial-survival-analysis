"""Recurrence-free survival analysis of the GBSG2 breast cancer cohort.

Pipeline (all outputs are written to ``outputs/`` and ``figures/``):

1. Kaplan-Meier estimates and log-rank tests by hormone therapy and tumor grade.
2. Cox proportional hazards model with ordinal grade (I=1, II=2, III=3).
3. Grambsch-Therneau score test of proportional hazards on log(time), which
   flags grade, so the primary model stratifies the baseline hazard by grade.
4. Harrell's C for both models, with a bootstrap optimism correction for the
   primary model.
5. Diagnostics: spline likelihood-ratio tests of linearity, null-model
   martingale residual plots and DFBETAS influence.
6. Sensitivity analysis: covariate-by-log(time) effects for the terms that
   still violate proportional hazards, fitted on counting-process data split
   at every event time (equivalent to R's ``coxph(..., tt = ...)``).

All Cox fits use Breslow ties so the Python, R and SAS implementations target
the same partial likelihood.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from patsy import dmatrix  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.duration.hazard_regression import PHReg  # noqa: E402
from statsmodels.duration.survfunc import SurvfuncRight, survdiff  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

ROOT = Path(__file__).resolve().parent
REQUIRED = ["horTh", "age", "menostat", "tsize", "tgrade", "pnodes", "progrec", "estrec", "time_days", "event"]
GRADE = {"I": 1, "II": 2, "III": 3}
CONTINUOUS = ["age", "tsize", "pnodes", "progrec", "estrec"]
# Reporting units for hazard ratios of continuous terms (same model, rescaled).
HR_UNITS = {"horTh_yes": (1, "yes vs no"), "age": (10, "per 10 years"),
            "menostat_Post": (1, "post vs pre"), "tsize": (10, "per 10 mm"),
            "grade_ord": (1, "per grade step"), "pnodes": (1, "per node"),
            "progrec": (100, "per 100 fmol/mg"), "estrec": (100, "per 100 fmol/mg")}
LABELS = {"horTh_yes": "Hormone therapy", "age": "Age", "menostat_Post": "Postmenopausal",
          "tsize": "Tumor size", "grade_ord": "Tumor grade", "pnodes": "Positive nodes",
          "progrec": "Progesterone receptor", "estrec": "Estrogen receptor"}
TIME_VARYING = ["age", "menostat_Post", "estrec"]
FIVE_YEARS = 5 * 365.25
SEED = 20261023
N_BOOT = 200


# --------------------------------------------------------------------------- data
def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load GBSG2 and check the schema and published counts (686 women, 299 events)."""
    df = pd.read_csv(Path(path) if path else ROOT / "data" / "gbsg2.csv")
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if len(df) != 686 or not set(df.event.unique()).issubset({0, 1}) or df[REQUIRED].isna().any().any():
        raise ValueError("unexpected GBSG2 schema/counts")
    return df


def design_matrix(df: pd.DataFrame, include_grade: bool = True) -> pd.DataFrame:
    """Numeric covariates; grade enters as an ordinal score only in the original model."""
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
    """Fit the original (ordinal grade) or primary (grade-stratified) Cox model."""
    X = design_matrix(df, include_grade=not stratified)
    model = PHReg(df.time_days, X, status=df.event, strata=df.tgrade.map(GRADE).to_numpy() if stratified else None, ties="breslow")
    return model.fit(disp=0), X


def coef_table(result, model_name: str) -> pd.DataFrame:
    """Per-unit and per-reporting-unit hazard ratios with Wald 95% CIs."""
    names = list(result.model.exog_names)
    b, se = np.asarray(result.params), np.asarray(result.bse)
    z = stats.norm.ppf(0.975)
    scale = np.array([HR_UNITS[n][0] for n in names], dtype=float)
    return pd.DataFrame({
        "model": model_name, "term": names, "log_hr": b, "se": se,
        "hr": np.exp(b), "ci_low": np.exp(b - z * se), "ci_high": np.exp(b + z * se),
        "p_value": np.asarray(result.pvalues),
        "report_unit": [HR_UNITS[n][1] for n in names],
        "hr_report": np.exp(scale * b), "ci_low_report": np.exp(scale * (b - z * se)),
        "ci_high_report": np.exp(scale * (b + z * se)),
    })


# --------------------------------------------------------------- Kaplan-Meier
def km_summary(df: pd.DataFrame, group: str | None) -> pd.DataFrame:
    """Median RFS (log-log CI) and 5-year RFS (log-log CI) overall or per group."""
    rows = []
    groups = [("all", df)] if group is None else [(g, d) for g, d in df.groupby(group, sort=True)]
    for level, d in groups:
        sf = SurvfuncRight(d.time_days.to_numpy(float), d.event.to_numpy(int))
        med = sf.quantile(0.5)
        lo, hi = sf.quantile_ci(0.5, method="cloglog")
        idx = np.searchsorted(sf.surv_times, FIVE_YEARS, side="right") - 1
        s5, se5 = float(sf.surv_prob[idx]), float(sf.surv_prob_se[idx])
        # log-log transformed pointwise CI for S(t), matching R's conf.type = "log-log".
        theta = np.log(-np.log(s5)); se_theta = se5 / (s5 * abs(np.log(s5)))
        z = stats.norm.ppf(0.975)
        rows.append({"group": group or "overall", "level": str(level), "n": len(d), "events": int(d.event.sum()),
                     "median_days": _nan_float(med), "median_ci_low": _nan_float(lo), "median_ci_high": _nan_float(hi),
                     "rfs_5y": s5, "rfs_5y_ci_low": float(np.exp(-np.exp(theta + z * se_theta))),
                     "rfs_5y_ci_high": float(np.exp(-np.exp(theta - z * se_theta)))})
    return pd.DataFrame(rows)


def _nan_float(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def logrank(df: pd.DataFrame, group: str) -> dict:
    chisq, p = survdiff(df.time_days.to_numpy(float), df.event.to_numpy(int), df[group].to_numpy())
    return {"group": group, "chisq": float(chisq), "df": int(df[group].nunique() - 1), "p_value": float(p)}


# ------------------------------------------------------------------ concordance
def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray, strata: np.ndarray | None = None) -> float:
    """Harrell's C, matching ``survival::concordance``: pairs are comparable when the
    shorter time is an event (an event before a censoring at the same time counts),
    pairs tied on an event time are excluded, and ties in risk score count one half.
    With strata, pairs are only formed within a stratum."""
    time, event, risk = map(np.asarray, (time, event, risk))
    strata = np.zeros(len(time)) if strata is None else np.asarray(strata)
    conc = disc = tied = 0.0
    for s in np.unique(strata):
        m = strata == s
        t, e, r = time[m], event[m].astype(bool), risk[m]
        ti, tj = t[:, None], t[None, :]
        comparable = e[:, None] & ((ti < tj) | ((ti == tj) & ~e[None, :]))
        ri, rj = r[:, None], r[None, :]
        conc += np.sum(comparable & (ri > rj))
        disc += np.sum(comparable & (ri < rj))
        tied += np.sum(comparable & (ri == rj))
    return float((conc + 0.5 * tied) / (conc + disc + tied))


def model_c(df: pd.DataFrame, params: np.ndarray, stratified: bool) -> float:
    lp = design_matrix(df, include_grade=not stratified).to_numpy() @ params
    return harrell_c(df.time_days.to_numpy(), df.event.to_numpy(), lp, df.tgrade.to_numpy() if stratified else None)


def optimism_corrected_c(df: pd.DataFrame, stratified: bool = True, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Harrell's bootstrap optimism correction for the apparent C-index."""
    rng = np.random.default_rng(seed)
    fit, _ = fit_ph(df, stratified)
    apparent = model_c(df, np.asarray(fit.params), stratified)
    optimism = []
    for _ in range(n_boot):
        boot = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        bfit, _ = fit_ph(boot, stratified)
        beta = np.asarray(bfit.params)
        optimism.append(model_c(boot, beta, stratified) - model_c(df, beta, stratified))
    opt = float(np.mean(optimism))
    return {"apparent": apparent, "optimism": opt, "corrected": apparent - opt, "n_boot": n_boot, "seed": seed}


# ----------------------------------------------------------- PH score test
def ph_score_test(df: pd.DataFrame, result, X: pd.DataFrame, strata: np.ndarray | None = None) -> pd.DataFrame:
    """Grambsch-Therneau score test for beta_j(t) = beta_j + theta_j * log(t).

    Same statistic as ``survival::cox.zph(fit, transform = "log")`` (survival >= 3.0):
    the score for theta at (beta_hat, 0) tested against the joint information of
    (beta, theta), computed with Breslow risk sets.
    """
    beta = np.asarray(result.params)
    x = X.to_numpy(float)
    t, d = df.time_days.to_numpy(float), df.event.to_numpy(int)
    strata = np.zeros(len(t)) if strata is None else np.asarray(strata)
    p = x.shape[1]
    U = np.zeros(2 * p)
    info = np.zeros((2 * p, 2 * p))
    w_all = np.exp(x @ beta)
    for s in np.unique(strata):
        m = strata == s
        ts, ds, xs, ws = t[m], d[m], x[m], w_all[m]
        for tk in np.unique(ts[ds == 1]):
            at_risk = ts >= tk
            dying = (ts == tk) & (ds == 1)
            w = ws[at_risk]; xr = xs[at_risk]
            xbar = w @ xr / w.sum()
            V = (xr * w[:, None]).T @ xr / w.sum() - np.outer(xbar, xbar)
            nk = dying.sum(); g = np.log(tk)
            resid = (xs[dying] - xbar).sum(axis=0)
            U[:p] += resid; U[p:] += g * resid
            info += nk * np.kron(np.array([[1.0, g], [g, g * g]]), V)
    rows = []
    for j, term in enumerate(X.columns):
        idx = list(range(p)) + [p + j]
        u = U[idx]; u[:p] = 0.0  # score for beta is zero at the MLE
        chisq = float(u @ np.linalg.solve(info[np.ix_(idx, idx)], u))
        rows.append({"term": term, "chisq": chisq, "df": 1, "p_value": float(stats.chi2.sf(chisq, 1))})
    u = U.copy(); u[:p] = 0.0
    chisq = float(u @ np.linalg.solve(info, u))
    rows.append({"term": "GLOBAL", "chisq": chisq, "df": p, "p_value": float(stats.chi2.sf(chisq, p))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- diagnostics
def _spline_design(df: pd.DataFrame, terms: list[str], spline_df: int = 3) -> pd.DataFrame:
    """Primary-model covariates with each listed term replaced by a natural cubic spline basis."""
    X = design_matrix(df, include_grade=False)
    for term in terms:
        basis = dmatrix(f"cr(x, df={spline_df}, constraints='center') - 1", {"x": df[term].to_numpy(float)}, return_type="dataframe")
        basis.columns = [f"{term}_s{i}" for i in range(basis.shape[1])]
        X = pd.concat([X.drop(columns=term), basis.set_index(X.index)], axis=1)
    return X


def _fit_stratified(df: pd.DataFrame, X: pd.DataFrame):
    return PHReg(df.time_days, X, status=df.event, strata=df.tgrade.map(GRADE).to_numpy(), ties="breslow").fit(disp=0)


def nonlinearity_tests(df: pd.DataFrame, spline_df: int = 3) -> pd.DataFrame:
    """Likelihood-ratio test of a natural cubic spline (df=3) vs a linear term, one covariate
    at a time, in the grade-stratified model."""
    base, _ = fit_ph(df, stratified=True)
    rows = []
    for term in CONTINUOUS:
        lr = 2 * (_fit_stratified(df, _spline_design(df, [term], spline_df)).llf - base.llf)
        rows.append({"term": term, "lr_chisq": float(lr), "df": spline_df - 1,
                     "p_value": float(stats.chi2.sf(lr, spline_df - 1))})
    return pd.DataFrame(rows)


def spline_fit(df: pd.DataFrame, terms: list[str]):
    """Grade-stratified model with natural cubic splines for ``terms``; returns (fit, X)."""
    X = _spline_design(df, terms)
    return _fit_stratified(df, X), X


def martingale_residuals(df: pd.DataFrame, lp: np.ndarray | None = None) -> np.ndarray:
    """Martingale residuals (event - exp(lp) * H0(t)) with a grade-stratified Breslow baseline.

    With ``lp=None`` this is the null model, where H0 is the Nelson-Aalen estimate.
    """
    lp = np.zeros(len(df)) if lp is None else np.asarray(lp)
    out = np.zeros(len(df))
    for _, pos in df.reset_index(drop=True).groupby(df.tgrade.to_numpy()).indices.items():
        t, e, r = df.time_days.to_numpy(float)[pos], df.event.to_numpy(int)[pos], np.exp(lp[pos])
        times = np.unique(t[e == 1])
        hazard = np.array([np.sum((t == u) & (e == 1)) / r[t >= u].sum() for u in times])
        cumhaz = np.concatenate([[0.0], np.cumsum(hazard)])
        out[pos] = e - r * cumhaz[np.searchsorted(times, t, side="right")]
    return out


def dfbetas(result, X: pd.DataFrame) -> pd.DataFrame:
    """Approximate scaled influence (score residuals x covariance / SE), as R's residuals(type = "dfbetas")."""
    score = np.asarray(result.score_residuals)
    dfb = score @ np.asarray(result.cov_params()) / np.asarray(result.bse)
    return pd.DataFrame(dfb, columns=X.columns)


# ---------------------------------------------- time-varying sensitivity
def split_at_event_times(df: pd.DataFrame) -> pd.DataFrame:
    """Counting-process (start, stop] data split at every observed event time."""
    cuts = np.unique(df.time_days[df.event == 1].to_numpy(float))
    rows = []
    for i, (t, e) in enumerate(zip(df.time_days.to_numpy(float), df.event.to_numpy(int))):
        stops = np.append(cuts[cuts < t], t)
        starts = np.concatenate([[0.0], stops[:-1]])
        status = np.zeros(len(stops), dtype=int)
        status[-1] = e
        rows.append(pd.DataFrame({"row": i, "start": starts, "stop": stops, "status": status}))
    return pd.concat(rows, ignore_index=True)


def fit_counting_process(df: pd.DataFrame, long: pd.DataFrame, X: pd.DataFrame):
    """Grade-stratified Cox fit on (start, stop] data.

    statsmodels counts a row as at risk at time t when entry <= t, while the
    counting-process convention (R's ``Surv(start, stop, event)``) is entry < t.
    Entry is therefore nudged forward by half the smallest gap between distinct
    times, which leaves every risk set exactly as R defines it.
    """
    gap = np.diff(np.unique(np.r_[0.0, df.time_days.to_numpy(float)])).min()
    strata = df.tgrade.map(GRADE).to_numpy()[long.row.to_numpy()]
    entry = np.where(long.start > 0, long.start + gap / 2, 0.0)
    return PHReg(long.stop, X, status=long.status, entry=entry, strata=strata, ties="breslow").fit(disp=0)


def time_varying_fit(df: pd.DataFrame) -> pd.DataFrame:
    """Grade-stratified Cox model adding x * log(t / 1 year) for the covariates that still
    fail the PH test. Main effects are therefore log-HRs at 1 year of follow-up."""
    long = split_at_event_times(df)
    X = design_matrix(df, include_grade=False).iloc[long.row.to_numpy()].reset_index(drop=True)
    log_t = np.log(long.stop.to_numpy() / 365.25)
    for term in TIME_VARYING:
        X[f"{term}_x_logt"] = X[term] * log_t
    fit = fit_counting_process(df, long, X)
    z = stats.norm.ppf(0.975)
    b, se = np.asarray(fit.params), np.asarray(fit.bse)
    return pd.DataFrame({"model": "grade_stratified_time_varying", "term": X.columns, "log_hr": b, "se": se,
                         "hr": np.exp(b), "ci_low": np.exp(b - z * se), "ci_high": np.exp(b + z * se),
                         "p_value": np.asarray(fit.pvalues), "loglik": float(fit.llf)})


# --------------------------------------------------------------------- plots
def _km_axis(ax, df: pd.DataFrame, group: str | None, label: str) -> None:
    groups = [("All patients", df)] if group is None else list(df.groupby(group, sort=True))
    for level, d in groups:
        sf = SurvfuncRight(d.time_days.to_numpy(float), d.event.to_numpy(int))
        name = level if group is None else f"{label} {level} (n={len(d)})"
        ax.step(np.r_[0, sf.surv_times] / 365.25, np.r_[1, sf.surv_prob], where="post", label=name)
    ax.set(xlabel="Years since study entry", ylabel="Recurrence-free survival", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower left")


def _forest_axis(ax, table: pd.DataFrame) -> None:
    t = table.iloc[::-1].reset_index(drop=True)
    ax.errorbar(t.hr_report, t.index, xerr=[t.hr_report - t.ci_low_report, t.ci_high_report - t.hr_report],
                fmt="o", color="#1f4e79", capsize=3)
    ax.axvline(1, color="grey", lw=0.8, ls="--")
    ax.set_yticks(t.index, [f"{LABELS[r.term]} ({r.report_unit})" for r in t.itertuples()])
    ax.set_xscale("log")
    ticks = [0.5, 0.7, 1.0, 1.4, 2.0]
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Hazard ratio (95% CI), log scale")
    ax.grid(axis="x", alpha=0.3)


def plot_km(df: pd.DataFrame, group: str | None, path: Path, title: str, label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _km_axis(ax, df, group, label)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_forest(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _forest_axis(ax, table)
    ax.set_title("Grade-stratified Cox model: recurrence-free survival")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_summary(df: pd.DataFrame, table: pd.DataFrame, path: Path) -> None:
    """Two-panel README figure: KM by hormone therapy and the primary-model forest plot."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    _km_axis(a, df, "horTh", "Hormone therapy:")
    a.set_title("Kaplan-Meier by hormone therapy")
    _forest_axis(b, table)
    b.set_title("Adjusted hazard ratios, grade-stratified Cox model")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_functional_form(df: pd.DataFrame, path: Path) -> None:
    resid = martingale_residuals(df)
    fig, axes = plt.subplots(1, len(CONTINUOUS), figsize=(15, 3.2), sharey=True)
    for ax, term in zip(axes, CONTINUOUS):
        x = df[term].to_numpy(float)
        ax.scatter(x, resid, s=6, alpha=0.3, color="grey")
        sm = lowess(resid, x, frac=0.6)
        ax.plot(sm[:, 0], sm[:, 1], color="#c0392b", lw=2)
        ax.set_xlabel(LABELS[term]); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Null-model martingale residual")
    fig.suptitle("Functional form check (lowess of martingale residuals; a straight line supports a linear term)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------- run
def run(data_path: str | Path | None = None, output_dir: str | Path | None = None,
        figure_dir: str | Path | None = None, n_boot: int = N_BOOT) -> dict:
    out = Path(output_dir) if output_dir else ROOT / "outputs"
    figs = Path(figure_dir) if figure_dir else (ROOT / "figures" if output_dir is None else out / "figures")
    (out / "figures").mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)

    km = pd.concat([km_summary(df, None), km_summary(df, "horTh"), km_summary(df, "tgrade")], ignore_index=True)
    km.to_csv(out / "km_summary.csv", index=False)
    lr = [logrank(df, "horTh"), logrank(df, "tgrade")]
    pd.DataFrame(lr).to_csv(out / "logrank_tests.csv", index=False)

    original, X0 = fit_ph(df, stratified=False)
    primary, X1 = fit_ph(df, stratified=True)
    original_table = coef_table(original, "original_ordinal_grade")
    primary_table = coef_table(primary, "primary_grade_stratified")
    pd.concat([original_table, primary_table], ignore_index=True).to_csv(out / "cox_coefficients.csv", index=False)

    ph_orig = ph_score_test(df, original, X0).assign(model="original_ordinal_grade")
    ph_prim = ph_score_test(df, primary, X1, df.tgrade.to_numpy()).assign(model="primary_grade_stratified")
    pd.concat([ph_orig, ph_prim], ignore_index=True).to_csv(out / "ph_diagnostics.csv", index=False)

    nonlin = nonlinearity_tests(df)
    nonlin.to_csv(out / "functional_form_diagnostics.csv", index=False)
    dfb = dfbetas(primary, X1)
    top = dfb.abs().max(axis=1).sort_values(ascending=False).head(10)
    influence = pd.DataFrame({"row_0based": top.index, "max_abs_dfbetas": top.values,
                              "term": dfb.abs().loc[top.index].idxmax(axis=1).values})
    influence.to_csv(out / "influence_diagnostics.csv", index=False)

    tv = time_varying_fit(df)
    tv.to_csv(out / "time_varying_sensitivity.csv", index=False)
    spline, Xs = spline_fit(df, CONTINUOUS)

    c_orig = model_c(df, np.asarray(original.params), stratified=False)
    c_prim = optimism_corrected_c(df, stratified=True, n_boot=n_boot)
    c_spline = harrell_c(df.time_days.to_numpy(), df.event.to_numpy(), Xs.to_numpy() @ np.asarray(spline.params),
                         df.tgrade.to_numpy())

    def horth(fit, names, label, c):
        j = list(names).index("horTh_yes")
        b, se = float(np.asarray(fit.params)[j]), float(np.asarray(fit.bse)[j])
        z = stats.norm.ppf(0.975)
        return {"model": label, "hr": np.exp(b), "ci_low": np.exp(b - z * se), "ci_high": np.exp(b + z * se),
                "p_value": float(np.asarray(fit.pvalues)[j]), "loglik": float(fit.llf), "c_index_apparent": c}

    tvx = tv.set_index("term").loc["horTh_yes"]
    sensitivity = pd.DataFrame([
        horth(original, X0.columns, "original: ordinal grade, all terms linear", c_orig),
        horth(primary, X1.columns, "primary: grade-stratified", c_prim["apparent"]),
        {"model": "primary + age, menopause, ER x log(t)", "hr": tvx.hr, "ci_low": tvx.ci_low,
         "ci_high": tvx.ci_high, "p_value": tvx.p_value, "loglik": tvx.loglik, "c_index_apparent": None},
        horth(spline, Xs.columns, "primary + natural splines for all continuous terms", c_spline),
    ])
    sensitivity.to_csv(out / "hormone_therapy_sensitivity.csv", index=False)

    contract = df.copy(); contract["grade_ord"] = contract.tgrade.map(GRADE)
    contract.to_csv(out / "analysis_input_contract.csv", index=False)

    hr_horth = primary_table.set_index("term").loc["horTh_yes"]
    metrics = {
        "n": int(len(df)), "events": int(df.event.sum()), "censored": int((df.event == 0).sum()),
        "grade_counts": df.tgrade.value_counts().sort_index().to_dict(),
        "hormone_counts": df.horTh.value_counts().sort_index().to_dict(),
        "estimand": "recurrence-free survival in days; event=recurrence or death; censored=0",
        "grade_coding": GRADE,
        "original_terms": list(X0.columns), "primary_terms": list(X1.columns),
        "logrank": {r["group"]: {k: r[k] for k in ("chisq", "df", "p_value")} for r in lr},
        "km_median_days": {f"{r.group}={r.level}": r.median_days for r in km.itertuples()},
        "km_rfs_5y": {f"{r.group}={r.level}": r.rfs_5y for r in km.itertuples()},
        "original_loglik": float(original.llf), "primary_loglik": float(primary.llf),
        "concordance": {"original_apparent": c_orig, "primary": c_prim},
        "primary_hormone_therapy_hr": {k: float(hr_horth[k]) for k in ("hr", "ci_low", "ci_high", "p_value")},
        "hormone_therapy_hr_range_across_models": [float(sensitivity.hr.min()), float(sensitivity.hr.max())],
        "ph_global_p": {"original": float(ph_orig.set_index("term").loc["GLOBAL", "p_value"]),
                        "primary": float(ph_prim.set_index("term").loc["GLOBAL", "p_value"])},
        "ph_flagged_primary": ph_prim.query("term != 'GLOBAL' and p_value < 0.05").term.tolist(),
        "nonlinear_terms": nonlin.query("p_value < 0.05").term.tolist(),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_km(df, None, figs / "km_overall.png", "Recurrence-free survival, GBSG2 (n=686)")
    plot_km(df, "horTh", figs / "km_hormone.png", "Recurrence-free survival by hormone therapy", "Hormone therapy:")
    plot_km(df, "tgrade", figs / "km_grade.png", "Recurrence-free survival by tumor grade", "Grade")
    plot_forest(primary_table, figs / "cox_forest.png")
    plot_summary(df, primary_table, figs / "summary.png")
    plot_functional_form(df, out / "figures" / "functional_form.png")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    lp = X1.to_numpy() @ np.asarray(primary.params)
    ax.scatter(lp, martingale_residuals(df, lp), s=8, alpha=0.45)
    ax.axhline(0, color="k", lw=0.7); ax.set(xlabel="Cox linear predictor (primary model)", ylabel="Martingale residual")
    fig.tight_layout(); fig.savefig(out / "figures" / "martingale_vs_lp.png", dpi=130); plt.close(fig)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data"); parser.add_argument("--output-dir"); parser.add_argument("--figure-dir")
    args = parser.parse_args()
    m = run(args.data, args.output_dir, args.figure_dir)
    print(json.dumps({k: m[k] for k in ("n", "events", "logrank", "concordance", "primary_hormone_therapy_hr",
                                        "ph_global_p", "nonlinear_terms")}, indent=2))
