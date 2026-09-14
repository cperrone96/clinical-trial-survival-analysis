# Reproducible GBSG2 analysis. Run from the repository root or pass
# --data /absolute/path/to/gbsg2.csv. Grade is I=1, II=2, III=3 everywhere.
suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
data_path <- if ("--data" %in% args) args[match("--data", args) + 1] else file.path(getwd(), "data", "gbsg2.csv")
out_dir <- if ("--output-dir" %in% args) args[match("--output-dir", args) + 1] else file.path(getwd(), "outputs-r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(data_path, stringsAsFactors = FALSE)
stopifnot(nrow(df) == 686, all(df$event %in% c(0, 1)))
df$horTh <- factor(df$horTh, levels = c("no", "yes"))
df$menostat <- factor(tolower(df$menostat), levels = c("pre", "post"))
df$grade_ord <- unname(c(I = 1, II = 2, III = 3)[df$tgrade])
df$tgrade <- factor(df$tgrade, levels = c("I", "II", "III"), ordered = TRUE)
surv_obj <- Surv(df$time_days, df$event)

# Original constant-HR grade estimand and the resolution: stratified baseline
# hazard by grade, with no grade HR reported.
original <- coxph(surv_obj ~ horTh + age + menostat + tsize + grade_ord + pnodes + progrec + estrec, data = df, ties = "breslow")
revised <- coxph(surv_obj ~ horTh + age + menostat + tsize + pnodes + progrec + estrec + strata(tgrade), data = df, ties = "breslow")

write.csv(as.data.frame(summary(original)$coefficients), file.path(out_dir, "cox_original_coefficients.csv"))
write.csv(as.data.frame(summary(revised)$coefficients), file.path(out_dir, "cox_revised_coefficients.csv"))
ph <- cox.zph(original)
write.csv(as.data.frame(ph$table), file.path(out_dir, "ph_diagnostics.csv"))
metrics <- data.frame(n = nrow(df), events = sum(df$event), censored = sum(df$event == 0),
                      original_concordance = summary(original)$concordance[1],
                      revised_concordance = summary(revised)$concordance[1])
write.csv(metrics, file.path(out_dir, "metrics.csv"), row.names = FALSE)
capture.output(summary(original), file = file.path(out_dir, "original_summary.txt"))
capture.output(summary(revised), file = file.path(out_dir, "revised_summary.txt"))
cat(sprintf("%d patients | %d events | grade PH test p=%.4g\n", nrow(df), sum(df$event), ph$table["grade_ord", "p"]))
