# GBSG2 data provenance

- **Source:** `TH.data::GBSG2`, CRAN package documentation version 1.1-2: <https://rdrr.io/cran/TH.data/man/GBSG2.html>
- **License:** GPL-3, confirmed from the CRAN DESCRIPTION file of TH.data 1.1-5 (<https://cran.r-project.org/web/packages/TH.data/DESCRIPTION>, checked 2026-10-27).
- **Original references:** Schumacher et al. (1994), *Journal of Clinical Oncology* 12:2086–2093; Sauerbrei & Royston (1999), *JRSSA* 162(1):71–94.
- **Local transformation:** `time` renamed `time_days`; `cens` renamed `event`; all rows, values, and factor labels retained.
- **Expected schema:** 686 rows; 299 events; 387 censored; 10 columns documented in `README.md`.
- **SHA-256:** `63a36b795dede63f0c8015010a65c9cd17c4593989a8b39172587135a231024e` for `data/gbsg2.csv`.

# Rotterdam data provenance

- **Source:** `survival::rotterdam` from the R `survival` package 3.5-7 (license LGPL >= 2), exported unchanged with `write.csv(survival::rotterdam, row.names = FALSE)`.
- **Reference:** Royston P, Altman DG (2013), *BMC Medical Research Methodology* 13:33.
- **Schema:** 2,982 rows, 15 columns; the external validation uses the 1,546 node-positive patients.
- **SHA-256:** `59598780fb4c4fd26a4a32289d514fbc499f7366b753ceb2b89bbe3a3db46cde` for `data/rotterdam.csv`.

