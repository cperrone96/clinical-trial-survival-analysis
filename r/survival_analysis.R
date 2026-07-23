# =============================================================================
# Survival analysis of the GBSG2 breast-cancer trial
# Kaplan-Meier estimation, log-rank tests, and Cox proportional-hazards
# regression using the {survival} and {survminer} packages -- the same workflow
# used to produce time-to-event figures for peer-reviewed publications.
# =============================================================================

# install.packages(c("survival", "survminer"))
library(survival)
library(survminer)

df <- read.csv("../data/gbsg2.csv", stringsAsFactors = TRUE)

# Time-to-event object: event = 1 (recurrence/death), 0 (censored)
surv_obj <- Surv(time = df$time_days, event = df$event)

# -----------------------------------------------------------------------------
# 1. Overall Kaplan-Meier recurrence-free survival
# -----------------------------------------------------------------------------
km_overall <- survfit(surv_obj ~ 1, data = df)
print(km_overall)                      # median survival + n at risk
summary(km_overall, times = c(365, 730, 1095, 1825))   # 1-, 2-, 3-, 5-year

ggsurvplot(km_overall, data = df, conf.int = TRUE,
           title = "Overall recurrence-free survival (Kaplan-Meier)",
           xlab = "Time (days)", ylab = "Survival probability")

# -----------------------------------------------------------------------------
# 2. Kaplan-Meier by hormonal therapy + log-rank test
# -----------------------------------------------------------------------------
km_hort <- survfit(surv_obj ~ horTh, data = df)
ggsurvplot(km_hort, data = df, pval = TRUE, conf.int = TRUE, risk.table = TRUE,
           legend.title = "Hormonal therapy", legend.labs = c("no", "yes"),
           title = "Recurrence-free survival by hormonal therapy",
           xlab = "Time (days)")
survdiff(surv_obj ~ horTh, data = df)  # log-rank test

# -----------------------------------------------------------------------------
# 3. Kaplan-Meier by tumor grade + multi-group log-rank test
# -----------------------------------------------------------------------------
km_grade <- survfit(surv_obj ~ tgrade, data = df)
ggsurvplot(km_grade, data = df, pval = TRUE,
           title = "Recurrence-free survival by tumor grade", xlab = "Time (days)")
survdiff(surv_obj ~ tgrade, data = df)

# -----------------------------------------------------------------------------
# 4. Cox proportional-hazards model -> adjusted hazard ratios
# -----------------------------------------------------------------------------
cox <- coxph(surv_obj ~ horTh + age + menostat + tsize + tgrade +
                        pnodes + progrec + estrec, data = df)
summary(cox)                           # hazard ratios, 95% CI, p-values
ggforest(cox, data = df)               # forest plot of hazard ratios

# -----------------------------------------------------------------------------
# 5. Proportional-hazards assumption (scaled Schoenfeld residuals)
# -----------------------------------------------------------------------------
ph <- cox.zph(cox)
print(ph)
ggcoxzph(ph)
