# GBSG2 recurrence-free survival analysis in R (survival package).
# Mirrors analysis.py so the two implementations can be compared term by term.
# Usage (from the repository root):
#   Rscript r/survival_analysis.R --data data/gbsg2.csv --output-dir outputs-r
suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
arg <- function(flag, default) if (flag %in% args) args[match(flag, args) + 1] else default
data_path <- arg("--data", file.path("data", "gbsg2.csv"))
out_dir <- arg("--output-dir", "outputs-r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(data_path, stringsAsFactors = FALSE)
stopifnot(nrow(df) == 686, all(df$event %in% c(0, 1)), !anyNA(df))
df$horTh <- factor(df$horTh, levels = c("no", "yes"))
df$menostat <- factor(tolower(df$menostat), levels = c("pre", "post"))
df$grade_ord <- unname(c(I = 1, II = 2, III = 3)[df$tgrade])
df$tgrade <- factor(df$tgrade, levels = c("I", "II", "III"))

# Kaplan-Meier and log-rank -------------------------------------------------
km_row <- function(fit, label) {
  s <- summary(fit)$table
  if (is.null(dim(s))) s <- t(as.matrix(s))
  five <- summary(fit, times = 5 * 365.25, extend = TRUE)
  data.frame(stratum = if (is.null(label)) rownames(s) else label,
             n = s[, "n.max"], events = s[, "events"], median_days = s[, "median"],
             median_ci_low = s[, "0.95LCL"], median_ci_high = s[, "0.95UCL"],
             rfs_5y = five$surv, rfs_5y_ci_low = five$lower, rfs_5y_ci_high = five$upper)
}
km <- rbind(
  km_row(survfit(Surv(time_days, event) ~ 1, df, conf.type = "log-log"), "overall"),
  km_row(survfit(Surv(time_days, event) ~ horTh, df, conf.type = "log-log"), NULL),
  km_row(survfit(Surv(time_days, event) ~ tgrade, df, conf.type = "log-log"), NULL))
write.csv(km, file.path(out_dir, "km_summary.csv"), row.names = FALSE)

lr <- lapply(c("horTh", "tgrade"), function(g) {
  sd <- survdiff(as.formula(paste("Surv(time_days, event) ~", g)), df)
  data.frame(group = g, chisq = sd$chisq, df = length(sd$n) - 1,
             p_value = pchisq(sd$chisq, length(sd$n) - 1, lower.tail = FALSE))
})
write.csv(do.call(rbind, lr), file.path(out_dir, "logrank_tests.csv"), row.names = FALSE)

# Cox models (Breslow ties to match statsmodels and PROC PHREG) -------------
original <- coxph(Surv(time_days, event) ~ horTh + age + menostat + tsize + grade_ord + pnodes + progrec + estrec,
                  data = df, ties = "breslow")
primary <- coxph(Surv(time_days, event) ~ horTh + age + menostat + tsize + pnodes + progrec + estrec + strata(tgrade),
                 data = df, ties = "breslow")
coef_out <- function(fit) {
  s <- summary(fit)
  data.frame(term = rownames(s$coefficients), log_hr = s$coefficients[, "coef"], se = s$coefficients[, "se(coef)"],
             hr = s$conf.int[, "exp(coef)"], ci_low = s$conf.int[, "lower .95"], ci_high = s$conf.int[, "upper .95"],
             p_value = s$coefficients[, "Pr(>|z|)"])
}
write.csv(coef_out(original), file.path(out_dir, "cox_original_coefficients.csv"), row.names = FALSE)
write.csv(coef_out(primary), file.path(out_dir, "cox_primary_coefficients.csv"), row.names = FALSE)

# Proportional hazards: Grambsch-Therneau score test on log(time) -----------
zph <- function(fit, model) {
  z <- cox.zph(fit, transform = "log")$table
  data.frame(model = model, term = rownames(z), chisq = z[, "chisq"], df = z[, "df"], p_value = z[, "p"])
}
write.csv(rbind(zph(original, "original_ordinal_grade"), zph(primary, "primary_grade_stratified")),
          file.path(out_dir, "ph_diagnostics.csv"), row.names = FALSE)

# Sensitivity: covariate x log(t / 1 year) for the terms still failing PH ----
tv <- coxph(Surv(time_days, event) ~ horTh + age + menostat + tsize + pnodes + progrec + estrec +
              tt(age) + tt(as.numeric(menostat == "post")) + tt(estrec) + strata(tgrade),
            data = df, ties = "breslow", tt = function(x, t, ...) x * log(t / 365.25))
write.csv(coef_out(tv), file.path(out_dir, "time_varying_sensitivity.csv"), row.names = FALSE)

metrics <- data.frame(n = nrow(df), events = sum(df$event), censored = sum(df$event == 0),
                      original_loglik = original$loglik[2], primary_loglik = primary$loglik[2],
                      time_varying_loglik = tv$loglik[2],
                      original_concordance = unname(concordance(original)$concordance),
                      primary_concordance = unname(concordance(primary)$concordance))
write.csv(metrics, file.path(out_dir, "metrics.csv"), row.names = FALSE)
capture.output(summary(original), file = file.path(out_dir, "original_summary.txt"))
capture.output(summary(primary), file = file.path(out_dir, "primary_summary.txt"))
cat(sprintf("%d patients | %d events | primary C = %.4f | hormone therapy HR = %.3f\n",
            nrow(df), sum(df$event), metrics$primary_concordance, exp(coef(primary)[["horThyes"]])))

# Functional form: natural cubic splines (df = 3) vs linear terms -----------
# splines::ns and patsy's cr place knots slightly differently from analysis.py,
# so these agree with the Python values closely but not to machine precision.
suppressPackageStartupMessages(library(splines))
continuous <- c("age", "tsize", "pnodes", "progrec", "estrec")
nonlin <- do.call(rbind, lapply(continuous, function(v) {
  fit <- update(primary, as.formula(paste(". ~ . -", v, "+ ns(", v, ", df = 3)")))
  lr <- 2 * (fit$loglik[2] - primary$loglik[2])
  data.frame(term = v, lr_chisq = lr, df = 2, p_value = pchisq(lr, 2, lower.tail = FALSE))
}))
write.csv(nonlin, file.path(out_dir, "functional_form_diagnostics.csv"), row.names = FALSE)
spline <- update(primary, . ~ . - age - tsize - pnodes - progrec - estrec + ns(age, df = 3) + ns(tsize, df = 3) +
                   ns(pnodes, df = 3) + ns(progrec, df = 3) + ns(estrec, df = 3))
write.csv(coef_out(spline), file.path(out_dir, "spline_sensitivity.csv"), row.names = FALSE)
