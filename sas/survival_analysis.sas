/* GBSG2 recurrence-free survival analysis in SAS.
   Mirrors analysis.py and r/survival_analysis.R: Kaplan-Meier, log-rank,
   original and grade-stratified Cox models (Breslow ties) and the
   covariate x log(t) sensitivity model.
   Set REPO_ROOT to the repository root before submitting.
   CONCORDANCE= requires SAS/STAT 15.1 (SAS 9.4M6) or later. */
%let REPO_ROOT=.;
%let DATA_PATH=&REPO_ROOT./data/gbsg2.csv;

proc import datafile="&DATA_PATH" out=gbsg2 dbms=csv replace;
    getnames=yes;
    guessingrows=max;
run;

data gbsg2;
    set gbsg2;
    /* Numeric indicators avoid any dependence on CLASS level formatting. */
    horTh_yes = (lowcase(horTh) = "yes");
    post_meno = (lowcase(menostat) = "post");
    /* Shared grade coding: I=1, II=2, III=3. */
    select (strip(tgrade));
        when ("I")   grade_ord = 1;
        when ("II")  grade_ord = 2;
        when ("III") grade_ord = 3;
        otherwise    grade_ord = .;
    end;
    label time_days="Recurrence-free time (days)" event="Event (1=recurrence or death, 0=censored)";
run;

/* Kaplan-Meier: log-log confidence limits and 5-year (1826.25 day) RFS, as in Python and R. */
proc lifetest data=gbsg2 conftype=loglog timelist=1826.25 reduceout outsurv=km_overall;
    time time_days*event(0);
run;
proc lifetest data=gbsg2 conftype=loglog timelist=1826.25;
    time time_days*event(0);
    strata horTh / test=logrank;
run;
proc lifetest data=gbsg2 conftype=loglog timelist=1826.25;
    time time_days*event(0);
    strata tgrade / test=logrank;
run;

/* Original model: ordinal grade with a constant hazard ratio. */
proc phreg data=gbsg2 concordance=harrell;
    model time_days*event(0) = horTh_yes age post_meno tsize grade_ord pnodes progrec estrec
          / ties=breslow risklimits;
    assess ph / resample seed=20261023;
    output out=cox_original xbeta=lp resmart=martingale
           ressch=sch_horTh sch_age sch_meno sch_tsize sch_grade sch_pnodes sch_progrec sch_estrec;
run;

/* Primary model: baseline hazard stratified by grade; no grade HR is estimated. */
proc phreg data=gbsg2 concordance=harrell;
    strata tgrade;
    model time_days*event(0) = horTh_yes age post_meno tsize pnodes progrec estrec
          / ties=breslow risklimits;
    output out=cox_primary xbeta=lp resmart=martingale dfbeta=dfb_horTh dfb_age dfb_meno dfb_tsize
           dfb_pnodes dfb_progrec dfb_estrec;
run;

/* Sensitivity: x * log(t / 1 year) for the covariates that still fail the PH test.
   Programming statements are evaluated at each event time, which is the
   counting-process construction used in Python and R's coxph(tt = ...). */
proc phreg data=gbsg2;
    strata tgrade;
    model time_days*event(0) = horTh_yes age post_meno tsize pnodes progrec estrec
          age_lt meno_lt estrec_lt / ties=breslow risklimits;
    age_lt    = age       * log(time_days / 365.25);
    meno_lt   = post_meno * log(time_days / 365.25);
    estrec_lt = estrec    * log(time_days / 365.25);
    nonph: test age_lt = 0, meno_lt = 0, estrec_lt = 0;
run;
