/* ==========================================================================
   Survival analysis of the GBSG2 breast-cancer trial
   PROC LIFETEST (Kaplan-Meier + log-rank) and PROC PHREG (Cox proportional
   hazards) -- the workflow used for clinical-trial statistical reporting.
   ========================================================================== */

proc import datafile="data/gbsg2.csv" out=gbsg2 dbms=csv replace;
    getnames=yes;
run;

/* Recode categorical predictors into numeric indicators / ordinal grade */
data gbsg2;
    set gbsg2;
    horTh_yes = (horTh = "yes");
    post_meno = (menostat = "Post");
    select (tgrade);
        when ("I")   grade_ord = 1;
        when ("II")  grade_ord = 2;
        when ("III") grade_ord = 3;
        otherwise    grade_ord = .;
    end;
    label time_days = "Recurrence-free time (days)"
          event     = "Event (1=recurrence/death, 0=censored)";
run;

/* --------------------------------------------------------------------------
   1. Overall Kaplan-Meier survival function
   -------------------------------------------------------------------------- */
proc lifetest data=gbsg2 plots=survival(atrisk cl);
    time time_days*event(0);           /* 0 = censoring value */
run;

/* --------------------------------------------------------------------------
   2. Kaplan-Meier by hormonal therapy, with log-rank test
   -------------------------------------------------------------------------- */
proc lifetest data=gbsg2 plots=survival(atrisk);
    time time_days*event(0);
    strata horTh / test=logrank;
run;

/* --------------------------------------------------------------------------
   3. Kaplan-Meier by tumor grade, with log-rank test
   -------------------------------------------------------------------------- */
proc lifetest data=gbsg2 plots=survival;
    time time_days*event(0);
    strata tgrade / test=logrank;
run;

/* --------------------------------------------------------------------------
   4. Cox proportional-hazards model -> adjusted hazard ratios (95% CI)
   -------------------------------------------------------------------------- */
proc phreg data=gbsg2 plots(overlay)=survival;
    class horTh (ref="no") menostat (ref="Pre") / param=ref;
    model time_days*event(0) = horTh age menostat tsize grade_ord
                               pnodes progrec estrec / risklimits;
    hazardratio horTh;
run;

/* --------------------------------------------------------------------------
   5. Proportional-hazards assumption check (Schoenfeld residuals)
   -------------------------------------------------------------------------- */
proc phreg data=gbsg2;
    model time_days*event(0) = horTh_yes age post_meno tsize grade_ord
                               pnodes progrec estrec;
    assess ph / resample;
run;
