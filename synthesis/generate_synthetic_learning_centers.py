"""
SYNTHETIC DATA GENERATOR - Learning-center visits (A5/A6/A7 group dataset)
==========================================================================
*** Every row this script produces is SYNTHETIC. None of it is a real student record. ***

Grounded in: learning_center_visits_spring2026_with_snacks.csv (40 real rows,
Spring 2026, Piedmont North + Commons).

Output: 720 rows = 18 centers x 2 simulated semesters x 10 week-periods x 2 time blocks
  C01-C09 = first-year halls (piedmont = 1), modeled on Piedmont North
  C10-C18 = upperclass halls (piedmont = 0), modeled on Commons

Usage:  python3 generate_synthetic_learning_centers.py real.csv
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SEED = 42
REAL_PATH = sys.argv[1] if len(sys.argv) > 1 else "real.csv"
OUT_PATH = "synthetic_learning_center_visits_720.csv"
CONTROLS = "+extended_hours+event_week+week_num+evening_block+midterm_week+spring_break"

# --------------------------------------------------------------------------
# STEP 0. Refit the real A3 model and the snack model on the real CSV
# --------------------------------------------------------------------------
real = pd.read_csv(REAL_PATH, dtype={"period": str})
FLY_MEAN, PIED_MEAN, SNACK_MEAN = (real.flyer_week.mean(), real.piedmont.mean(),
                                   real.snack_week.mean())          # 0.30, 0.50, 0.20


def add_centered(df):
    df = df.copy()
    df["x1"] = df.flyer_week - FLY_MEAN          # centered with the REAL means,
    df["x2"] = df.piedmont - PIED_MEAN           # so synthetic and real coefficients
    df["x1x2"] = df.x1 * df.x2                   # mean the same thing
    df["sc"] = df.snack_week - SNACK_MEAN
    df["scx2"] = df.sc * df.x2
    return df


real = add_centered(real)
a3_real = smf.ols("visits ~ x1 + x2 + x1x2" + CONTROLS, real).fit()
dgp_fit = smf.ols("visits ~ x1 + x2 + x1x2 + sc + scx2" + CONTROLS, real).fit()

# extended_hours == midterm_week + spring_break on every row, so the three are
# not separately identified. Fold extended_hours into the other two (fitted
# values are identical; only the bookkeeping changes).
assert (real.extended_hours == real.midterm_week + real.spring_break).all()

# --------------------------------------------------------------------------
# STEP 1. THE DATA-GENERATING PROCESS (all numbers come from dgp_fit above)
# --------------------------------------------------------------------------
p = dgp_fit.params
B = {
    "intercept":      p["Intercept"],                          # 17.081
    "x1_flyer":       p["x1"],                                 #  2.667
    "x2_piedmont":    p["x2"],                                 # 15.706
    "x1x2_flyer_x_piedmont": p["x1x2"],                        # -0.313  (see note A)
    # EXPLORATORY, POST-HOC (team-approved): snack terms were added after the
    # Section 1 residuals pointed at snack weeks. Not a pre-registered effect.
    "sc_snack":       p["sc"],                                 #  7.542
    "scx2_snack_x_piedmont": p["scx2"],                        #  9.937
    "event_week":     p["event_week"],                         # 20.187
    "week_num":       p["week_num"],                           #  0.824
    "evening_block":  p["evening_block"],                      # -5.555
    "midterm_week":   p["midterm_week"] + p["extended_hours"], #  0.647 (ext. hours folded in)
    "spring_break":   p["spring_break"] + p["extended_hours"], # -3.397 (ext. hours folded in)
}
# Note A: once snack terms are in the model, the structural flyer x piedmont
# interaction is only -0.31. The A3 value of -3.625 is mostly snack weeks being
# absorbed by the flyer interaction (an omitted-variable effect). Because the
# synthetic data keeps the real calendar, refitting the A3 model (no snack terms)
# on it reproduces that absorption, so A3's beta3 should again come out near -3.6.

# Noise: residual SD of THIS model by hall type (ddof = 1).
# The A3-model residual SDs (8.71 / 4.55) still contain the snack signal; using
# them on top of explicit snack terms would count that variation twice.
res = dgp_fit.resid
NOISE_SD = {1: res[real.piedmont == 1].std(),    # first-year halls  ~7.44
            0: res[real.piedmont == 0].std()}    # upperclass halls  ~4.32

# --------------------------------------------------------------------------
# STEP 2. Build the 720-row design using the real Spring 2026 calendar
# --------------------------------------------------------------------------
CALENDAR = pd.DataFrame({
    "period":       ["1-2", "3", "4", "5", "6", "7", "8-9", "10", "11", "12"],
    "week_num":     [1.5, 3, 4, 5, 6, 7, 8.5, 10, 11, 12],
    "flyer_week":   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "snack_week":   [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    "midterm_week": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "spring_break": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "extended_hours": [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
})
rows = []
for c in range(1, 19):
    pied = 1 if c <= 9 else 0
    for sem in ("Sim-Semester-1", "Sim-Semester-2"):
        for _, w in CALENDAR.iterrows():
            for block in ("day", "evening"):
                r = w.to_dict()
                r.update(center=f"C{c:02d}", hall_type="first-year" if pied else "upperclass",
                         semester=sem, block=block, piedmont=pied,
                         evening_block=int(block == "evening"),
                         event_week=int(pied == 1 and w.period == "7"))
                rows.append(r)
syn = add_centered(pd.DataFrame(rows))

# --------------------------------------------------------------------------
# STEP 3. Draw visits = structural mean + heteroskedastic normal noise
# --------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
mu = (B["intercept"] + B["x1_flyer"] * syn.x1 + B["x2_piedmont"] * syn.x2
      + B["x1x2_flyer_x_piedmont"] * syn.x1x2
      + B["sc_snack"] * syn.sc + B["scx2_snack_x_piedmont"] * syn.scx2       # exploratory
      + B["event_week"] * syn.event_week + B["week_num"] * syn.week_num
      + B["evening_block"] * syn.evening_block
      + B["midterm_week"] * syn.midterm_week + B["spring_break"] * syn.spring_break)
sd = syn.piedmont.map(NOISE_SD)
raw = mu + rng.normal(0.0, sd)
n_negative = int((raw < 0).sum())
syn["visits"] = raw.clip(lower=0).round(1)
syn["data_source"] = "synthetic"

COLS = ["data_source", "center", "hall_type", "semester", "period", "block", "visits",
        "flyer_week", "piedmont", "snack_week", "evening_block", "week_num",
        "event_week", "midterm_week", "spring_break", "extended_hours"]
syn[COLS].to_csv(OUT_PATH, index=False)

# --------------------------------------------------------------------------
# STEP 4. Verification
# --------------------------------------------------------------------------
pd.set_option("display.width", 140)
print("=" * 70)
print("SYNTHETIC DATA - seed", SEED, "-", len(syn), "rows written to", OUT_PATH)
print("=" * 70)
print("\n[0] Real A3 refit:  beta3 = %.3f  p = %.3f  95%% CI [%.2f, %.2f]" % (
    a3_real.params.x1x2, a3_real.pvalues.x1x2, *a3_real.conf_int().loc["x1x2"]))
print(a3_real.params.round(3).to_string())
print("\nDGP coefficients (real model + exploratory snack terms):")
for k, v in B.items():
    print(f"  {k:24s} {v:8.3f}")
print("  noise SD first-year %.2f, upperclass %.2f" % (NOISE_SD[1], NOISE_SD[0]))
print(f"\nNegative draws set to 0: {n_negative} of {len(syn)} rows")

FEATS = ["visits", "flyer_week", "piedmont", "snack_week", "evening_block", "week_num",
         "event_week", "midterm_week", "spring_break", "extended_hours"]
tab = pd.DataFrame({
    "real_mean": real[FEATS].mean(), "syn_mean": syn[FEATS].mean(),
    "real_sd": real[FEATS].std(), "syn_sd": syn[FEATS].std(),
    "real_r_visits": real[FEATS].corr()["visits"], "syn_r_visits": syn[FEATS].corr()["visits"],
}).round(3)
print("\n[1] Means, SDs and correlation with visits")
print(tab.to_string())
diff = (syn[FEATS].corr() - real[FEATS].corr()).abs()
print("\nFull correlation matrix: largest |synthetic - real| = %.3f (%s vs %s)" % (
    diff.values.max(), *diff.stack().idxmax()))

a3_syn = smf.ols("visits ~ x1 + x2 + x1x2" + CONTROLS, syn).fit()
full_syn = smf.ols("visits ~ x1 + x2 + x1x2 + sc + scx2" + CONTROLS, syn).fit()
print("\n[2] A3 model refit on synthetic: beta3 = %.3f  p = %.4f  95%% CI [%.2f, %.2f]" % (
    a3_syn.params.x1x2, a3_syn.pvalues.x1x2, *a3_syn.conf_int().loc["x1x2"]))
fy = full_syn.params.sc + full_syn.params.scx2 * (1 - PIED_MEAN)
uc = full_syn.params.sc + full_syn.params.scx2 * (0 - PIED_MEAN)
print("[3] Snack-week effect (exploratory model): first-year %+.2f, upperclass %+.2f, "
      "gap %.2f (p = %.4f)" % (fy, uc, fy - uc, full_syn.pvalues.scx2))
print("    Real:                                   first-year %+.2f, upperclass %+.2f, gap %.2f" % (
    p.sc + p.scx2 * 0.5, p.sc - p.scx2 * 0.5, p.scx2))
