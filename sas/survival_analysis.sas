/* GBSG2 analysis. Set REPO_ROOT before submitting, or edit DATA_PATH. */
%let REPO_ROOT=.;
%let DATA_PATH=&REPO_ROOT./data/gbsg2.csv;

proc import datafile="&DATA_PATH" out=gbsg2 dbms=csv replace;
    getnames=yes;
run;

data gbsg2;
    set gbsg2;
    horTh_yes = (lowcase(horTh) = "yes");
    post_meno = (lowcase(menostat) = "post");
    /* Shared grade contract: I=1, II=2, III=3; I is the reference. */
    if tgrade="I" then grade_ord=1;
    else if tgrade="II" then grade_ord=2;
    else if tgrade="III" then grade_ord=3;
    log_time=log(time_days/365.25);
    age_logtime=age*log_time;
    menostat_Post_logtime=post_meno*log_time;
    estrec_logtime=estrec*log_time;
    label time_days="Recurrence-free time (days)" event="Event (1=event, 0=censored)";
run;

proc lifetest data=gbsg2 outsurv=km_overall;
    time time_days*event(0);
run;
proc lifetest data=gbsg2;
    time time_days*event(0);
    strata horTh / test=logrank;
run;
proc lifetest data=gbsg2;
    time time_days*event(0);
    strata tgrade / test=logrank;
run;

/* Original constant-HR model, equivalent grade ordinal coding. */
proc phreg data=gbsg2;
    class horTh(ref="no") menostat(ref="pre") / param=ref;
    model time_days*event(0)=horTh age menostat tsize grade_ord pnodes progrec estrec / ties=breslow risklimits;
    assess ph / resample seed=20260914;
    output out=cox_original xbeta=lp resmart=martingale ressch=sch_grade;
run;

/* Primary revision: grade-stratified baseline hazard; no grade HR is reported. */
proc phreg data=gbsg2;
    class horTh(ref="no") menostat(ref="pre") / param=ref;
    strata tgrade;
    model time_days*event(0)=horTh age menostat tsize pnodes progrec estrec / ties=breslow risklimits;
    output out=cox_revised xbeta=lp;
run;

/* Sensitivity model for age, menopausal status, and ER trends flagged by the
   original Schoenfeld screen. Time is centered at one year. */
proc phreg data=gbsg2;
    class horTh(ref="no") menostat(ref="pre") / param=ref;
    strata tgrade;
    model time_days*event(0)=horTh age menostat tsize pnodes progrec estrec
        age_logtime menostat_Post_logtime estrec_logtime / ties=breslow risklimits;
    output out=cox_sensitivity xbeta=lp;
run;
