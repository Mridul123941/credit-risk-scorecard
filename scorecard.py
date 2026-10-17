"""
Credit-risk scorecard + monitoring layer on Kaggle's "Give Me Some Credit".

Pipeline
  1. Clean data, split into an "old" (development) and "new" (monitoring) sample
  2. Quantile / manual binning -> WoE and IV (fit on training data only)
  3. Logistic regression (statsmodels) on WoE features, VIF pruning, sign checks
  4. Evaluate: AUC, Gini, KS on train / held-out test
  5. Monitor on the "new" sample: PSI (score + features), AUC/Gini drop, calibration

Usage
  python scorecard.py --data data/cs-training.csv
Outputs go to outputs/<scenario>/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 42
TARGET = "default"
AGE_CUTOFF = 45  # proxy split for the "age_shift" scenario

RENAME = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "dpd30",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "income",
    "NumberOfOpenCreditLinesAndLoans": "open_lines",
    "NumberOfTimes90DaysLate": "dpd90",
    "NumberRealEstateLoansOrLines": "re_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "dpd60",
    "NumberOfDependents": "dependents",
}
FEATURES = ["util", "age", "dpd30", "debt_ratio", "income", "open_lines",
            "dpd90", "re_loans", "dpd60", "dependents"]
INF = np.inf
# Count-like variables get hand-picked bins; the rest get decile bins.
MANUAL_EDGES = {
    "dpd30": [-INF, 0, 1, 2, INF],
    "dpd60": [-INF, 0, 1, INF],
    "dpd90": [-INF, 0, 1, 2, INF],
    "re_loans": [-INF, 0, 1, 2, INF],
    "dependents": [-INF, 0, 1, 2, INF],
}


# ----------------------------------------------------------------- data
def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.rename(columns=RENAME)[[TARGET] + FEATURES]
    n0 = len(df)
    df = df[df["age"] >= 18].copy()  # drop impossible ages (age 0)
    # 96 / 98 in the delinquency counts are data-entry codes, not real counts.
    # Capping at 13 keeps those rows in the "very delinquent" bucket.
    for c in ["dpd30", "dpd60", "dpd90"]:
        df[c] = df[c].clip(upper=13)
    # Outliers in util / debt_ratio / income are handled by quantile binning
    # (extreme values just fall into the top bin). Missing income and
    # dependents are kept as their own "Missing" bin instead of being imputed.
    print(f"Rows: {n0} -> {len(df)} after cleaning")
    print(f"Default rate: {df[TARGET].mean():.2%}  (imbalanced)")
    print("Missing share:\n", df[FEATURES].isna().mean().round(3)[lambda s: s > 0])
    return df.reset_index(drop=True)


def split_population(df, scenario):
    if scenario == "control":  # no real drift expected
        cut = int(0.7 * len(df))
        return df.iloc[:cut], df.iloc[cut:]
    if scenario == "age_shift":  # simulated population drift
        return df[df["age"] >= AGE_CUTOFF], df[df["age"] < AGE_CUTOFF]
    raise ValueError(scenario)


# -------------------------------------------------------------- binning
def fit_edges(train):
    edges = {}
    for v in FEATURES:
        if v in MANUAL_EDGES:
            edges[v] = MANUAL_EDGES[v]
        else:
            q = train[v].dropna().quantile(np.linspace(0, 1, 11)).unique()
            edges[v] = np.concatenate([[-INF], q[1:-1], [INF]])
    return edges


def to_labels(s, edges):
    cat = pd.cut(s, bins=edges)
    labels = [str(i) for i in cat.cat.categories]
    lab = cat.cat.rename_categories(labels).astype(object)
    lab[s.isna()] = "Missing"
    return lab.astype(str), labels + ["Missing"]


def woe_table(lab, y, order):
    g = (pd.DataFrame({"bin": lab.values, "y": y.values})
         .groupby("bin")["y"].agg(n="count", bad="sum"))
    g = g.reindex([o for o in order if o in g.index])
    g["good"] = g["n"] - g["bad"]
    g["bad_rate"] = g["bad"] / g["n"]
    # +0.5 smoothing avoids log(0) in sparse bins
    g["pct_bad"] = (g["bad"] + 0.5) / (g["bad"].sum() + 0.5 * len(g))
    g["pct_good"] = (g["good"] + 0.5) / (g["good"].sum() + 0.5 * len(g))
    # WoE = ln(%bad / %good): higher WoE = riskier, so coefficients should be > 0
    g["woe"] = np.log(g["pct_bad"] / g["pct_good"])
    g["iv_part"] = (g["pct_bad"] - g["pct_good"]) * g["woe"]
    return g


def monotonic_rho(tab):
    t = tab.drop("Missing", errors="ignore")
    if len(t) < 3:
        return np.nan
    return spearmanr(range(len(t)), t["bad_rate"])[0]


def apply_woe(df, edges, tables):
    X = pd.DataFrame(index=df.index)
    for v in FEATURES:
        lab, _ = to_labels(df[v], edges[v])
        X[v] = lab.map(tables[v]["woe"]).fillna(0.0)
    return X


# ---------------------------------------------------------------- model
def prune_vif(X, thr=5.0):
    cols = list(X.columns)
    while len(cols) > 2:
        Xc = sm.add_constant(X[cols])
        vifs = pd.Series([variance_inflation_factor(Xc.values, i)
                          for i in range(1, Xc.shape[1])], index=cols)
        if vifs.max() <= thr:
            break
        print(f"  dropping {vifs.idxmax()} (VIF={vifs.max():.1f})")
        cols.remove(vifs.idxmax())
    Xc = sm.add_constant(X[cols])
    vifs = pd.Series([variance_inflation_factor(Xc.values, i)
                      for i in range(1, Xc.shape[1])], index=cols)
    return cols, vifs


def fit_logit(y, X, cols):
    while True:
        model = sm.Logit(y, sm.add_constant(X[cols])).fit(disp=0)
        wrong = [c for c in cols if model.params[c] <= 0]
        if not wrong:
            return model, cols
        worst = min(wrong, key=lambda c: model.params[c])
        print(f"  wrong-sign coefficient on {worst}; dropping and refitting")
        cols = [c for c in cols if c != worst]


def predict(model, X, cols):
    return model.predict(sm.add_constant(X[cols], has_constant="add"))


def metrics(y, p):
    auc = roc_auc_score(y, p)
    ks = ks_2samp(p[y == 1], p[y == 0]).statistic
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": float(ks),
            "mean_pd": float(np.mean(p)), "obs_default_rate": float(np.mean(y)),
            "n": int(len(y))}


# ----------------------------------------------------------- monitoring
def psi_labels(exp, act, order):
    e = exp.value_counts(normalize=True).reindex(order).fillna(0).clip(lower=1e-4)
    a = act.value_counts(normalize=True).reindex(order).fillna(0).clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def calibration_table(y, p, n=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bucket"] = pd.qcut(d["p"], n, duplicates="drop")
    t = d.groupby("bucket", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    return t.reset_index(drop=True)


def psi_flag(x):
    return "stable" if x < 0.10 else ("moderate shift" if x < 0.25 else "significant shift")


# ---------------------------------------------------------------- plots
def plot_roc(sets, path):
    plt.figure(figsize=(5, 4))
    for name, (y, p) in sets.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_calibration(tabs, path):
    plt.figure(figsize=(5, 4))
    for name, t in tabs.items():
        plt.plot(t["mean_pred"], t["observed"], marker="o", label=name)
    m = max(t[["mean_pred", "observed"]].max().max() for t in tabs.values())
    plt.plot([0, m], [0, m], "k--", lw=0.8)
    plt.xlabel("Mean predicted PD"); plt.ylabel("Observed default rate")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_psi(psi_df, path):
    s = psi_df["psi"].sort_values()
    plt.figure(figsize=(6, 4))
    plt.barh(s.index, s.values)
    plt.axvline(0.1, color="orange", ls="--"); plt.axvline(0.25, color="red", ls="--")
    plt.xlabel("PSI (new vs development)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ------------------------------------------------------------------ run
def run(df, scenario, root):
    out = root / scenario
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nScenario: {scenario}\n{'=' * 60}")

    old, new = split_population(df, scenario)
    # Stratified split keeps the rare-default share identical in train and test.
    train, test = train_test_split(old, test_size=0.3, stratify=old[TARGET],
                                   random_state=SEED)
    print(f"train={len(train)} test={len(test)} new={len(new)} | default rates "
          f"{train[TARGET].mean():.2%} / {test[TARGET].mean():.2%} / {new[TARGET].mean():.2%}")

    # --- WoE / IV (train only) ---
    edges = fit_edges(train)
    tables, rows = {}, {}
    for v in FEATURES:
        lab, order = to_labels(train[v], edges[v])
        tables[v] = woe_table(lab, train[TARGET], order)
        rows[v] = {"IV": tables[v]["iv_part"].sum(), "monotonic_rho": monotonic_rho(tables[v])}
    iv_df = pd.DataFrame(rows).T.sort_values("IV", ascending=False)
    print("\nInformation value:\n", iv_df.round(3))
    iv_df.to_csv(out / "iv_table.csv")
    pd.concat(tables, names=["variable", "bin"]).to_csv(out / "woe_bins.csv")

    # --- model ---
    cand = iv_df.index[iv_df["IV"] >= 0.02].tolist()
    X = {k: apply_woe(d, edges, tables) for k, d in
         {"train": train, "test": test, "new": new}.items()}
    print("\nVIF pruning:")
    cols, vifs = prune_vif(X["train"][cand])
    model, cols = fit_logit(train[TARGET], X["train"], cols)
    print(model.summary())
    (out / "model_summary.txt").write_text(str(model.summary()))
    vifs.to_csv(out / "vif.csv", header=["vif"])

    # --- evaluation ---
    y = {"train": train[TARGET].values, "test": test[TARGET].values, "new": new[TARGET].values}
    p = {k: predict(model, X[k], cols).values for k in X}
    res = {k: metrics(y[k], p[k]) for k in X}
    print("\nPerformance:\n", pd.DataFrame(res).T.round(4))

    # --- monitoring: PSI on score and features ---
    q = np.quantile(p["train"], np.linspace(0, 1, 11))
    s_edges = np.unique(np.concatenate([[-INF], q[1:-1], [INF]]))
    lab_tr, order_s = to_labels(pd.Series(p["train"]), s_edges)
    psi_rows = {"SCORE (test vs train)": psi_labels(
        lab_tr, to_labels(pd.Series(p["test"]), s_edges)[0], order_s),
        "SCORE (new vs train)": psi_labels(
        lab_tr, to_labels(pd.Series(p["new"]), s_edges)[0], order_s)}
    for v in FEATURES:
        l_tr, order = to_labels(train[v], edges[v])
        l_new, _ = to_labels(new[v], edges[v])
        psi_rows[v] = psi_labels(l_tr, l_new, order)
    psi_df = pd.DataFrame({"psi": psi_rows})
    psi_df["flag"] = psi_df["psi"].map(psi_flag)
    print("\nPSI:\n", psi_df.round(3))
    psi_df.to_csv(out / "psi.csv")

    # --- calibration ---
    cal = {"test": calibration_table(y["test"], p["test"]),
           "new": calibration_table(y["new"], p["new"])}
    for k, t in cal.items():
        t.to_csv(out / f"calibration_{k}.csv", index=False)
    print(f"\nCalibration: new sample mean PD {res['new']['mean_pd']:.2%} "
          f"vs observed {res['new']['obs_default_rate']:.2%}")

    # --- plots + summary ---
    plot_roc({k: (y[k], p[k]) for k in X}, out / "roc.png")
    plot_calibration(cal, out / "calibration.png")
    plot_psi(psi_df.drop(index=["SCORE (test vs train)", "SCORE (new vs train)"]),
             out / "feature_psi.png")
    summary = {"scenario": scenario, "features_in_model": cols,
               "coefficients": {k: float(v) for k, v in model.params.items()},
               "performance": res,
               "score_psi_new_vs_train": psi_rows["SCORE (new vs train)"],
               "gini_drop_test_to_new": res["test"]["gini"] - res["new"]["gini"]}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cs-training.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    df = load_and_clean(args.data)
    root = Path(args.out)
    summaries = [run(df, s, root) for s in ["control", "age_shift"]]
    print("\n" + "=" * 60 + "\nSUMMARY")
    for s in summaries:
        print(f"{s['scenario']:>10}: test Gini={s['performance']['test']['gini']:.3f} | "
              f"new Gini={s['performance']['new']['gini']:.3f} | "
              f"score PSI={s['score_psi_new_vs_train']:.3f} | "
              f"mean PD {s['performance']['new']['mean_pd']:.2%} vs "
              f"observed {s['performance']['new']['obs_default_rate']:.2%}")


if __name__ == "__main__":
    main()
