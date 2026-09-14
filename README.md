# Clinical-Trial Survival Analysis (GBSG2)

This reproducible educational analysis estimates recurrence-free survival (days to recurrence/death; censoring indicator `event=0`) in 686 women from the public **German Breast Cancer Study Group 2 (GBSG2)** data. The observation unit is one participant. The analysis is descriptive/associational for this archived dataset; it is not clinical validation, treatment guidance, or a claim about current patients.

## Data contract and provenance

The checked-in `data/gbsg2.csv` is a normalized copy of `TH.data::GBSG2` (CRAN package version 1.1-2 documentation, [dataset reference](https://rdrr.io/cran/TH.data/man/GBSG2.html)). The source describes 686 women, recurrence-free time, censoring, and the original GBSG publications (Schumacher et al., 1994; Sauerbrei & Royston, 1999). Transformation: column names `time` → `time_days` and `cens` → `event`; factor labels are retained; row order and values are otherwise unchanged. SHA-256: `63a36b795dede63f0c8015010a65c9cd17c4593989a8b39172587135a231024e`.

The single grade estimand is ordinal tumor grade: I=1, II=2, III=3. This coding is applied in Python, R, and SAS. The original model treats grade as a constant hazard-ratio term. A Schoenfeld residual time-trend diagnostic flags the grade term, so the primary revision stratifies the baseline hazard by grade and does not report a grade HR. Other covariates are baseline hormone assignment, age, menopausal status, tumor size, positive nodes, progesterone receptor, and estrogen receptor.

## Results and interpretation

The dataset contains 299 observed events and 387 censored records. Hormone-group and grade-group Kaplan–Meier/log-rank summaries are descriptive contrasts. The adjusted Cox estimates are associations conditional on the listed covariates and model assumptions; they do not establish that hormonal therapy caused a benefit. Residual, functional-form, influence, and proportional-hazards diagnostics are exported so readers can inspect model limitations.

## Implementations

- `analysis.py` is the deterministic Python analysis (`statsmodels` PHReg), writing `outputs/metrics.json`, coefficient/parity tables, diagnostics, and a residual plot.
- `r/survival_analysis.R` uses `survival::coxph`; pass `--data` and `--output-dir` explicitly for any working directory.
- `sas/survival_analysis.sas` uses `PROC LIFETEST` and `PROC PHREG`; set `REPO_ROOT` to the repository root before submitting. The original and grade-stratified models are both included.
- `survival_analysis.ipynb` is a clean, executed wrapper around `analysis.py`.

```bash
python -m pip install -r requirements.txt
python analysis.py
pytest -q
# Optional R parity (from repository root):
Rscript r/survival_analysis.R --data "$PWD/data/gbsg2.csv" --output-dir "$PWD/outputs-r"
```

The analysis is a public-data methods portfolio artifact. It contains no patient-identifying information, does not reproduce a sponsor workflow, and should not be used for treatment decisions.
