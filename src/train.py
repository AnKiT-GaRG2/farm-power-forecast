"""Train SVR, Decision Tree and an SVR+DT ensemble to forecast daily power
consumption (MU) of agrarian Indian states. Saves models, metrics and figures."""
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "long_data_.csv"
OUT = ROOT / "models"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# Agriculture-dominated states (large irrigation pumpset load)
AGRI_STATES = ["Punjab", "Haryana", "Uttar Pradesh", "Rajasthan", "Madhya Pradesh",
               "Maharashtra", "Gujarat", "Karnataka", "Andhra Pradesh", "Telangana",
               "Bihar", "Chhattisgarh", "Tamil Nadu", "West Bengal", "Odisha"]
ALIAS = {"UP": "Uttar Pradesh", "MP": "Madhya Pradesh"}
NUM = ["lag1", "lag2", "lag7", "roll7", "dow", "month", "doy_sin", "doy_cos"]
CAT = ["state"]
SEED = 42


def fix_dates(raw, usage, pos):
    """Repair day/month swaps in the source CSV (Excel locale issue).
    1) 35 trailing rows (true dates 6-12 Jan-May 2020) are stored as mm/dd and parse as
       Jun-Dec 2020; the data ends on 23 May 2020, so later dates are re-parsed as mm/dd.
    2) The 7th of Aug-Dec 2019 is stored as mm/dd ("08/07"..."12/07") and collides with
       8-12 Jul 2019. Each colliding pair is assigned by continuity of the all-state total
       with the previous day of each candidate date. `pos` = row position within a state."""
    dt = pd.to_datetime(raw, format="%d/%m/%Y %H:%M:%S")
    bad = dt > pd.Timestamp("2020-05-23")
    dt[bad] = pd.to_datetime(raw[bad], format="%m/%d/%Y %H:%M:%S")
    tot = usage.groupby(pos).sum()                      # national total per file row
    pdt = dt.groupby(pos).first()                       # date per file row
    for day in pd.date_range("2019-07-08", "2019-07-12"):
        rows = pdt.index[pdt == day]
        if len(rows) != 2:
            continue
        alt = pd.Timestamp(year=2019, month=day.day, day=day.month)  # swapped reading: 8 Jul -> 7 Aug
        ref = {k: tot[pdt.index[pdt == k - pd.Timedelta(days=1)][0]] for k in (day, alt)}
        r1, r2 = rows
        cost_keep = abs(tot[r1] - ref[day]) + abs(tot[r2] - ref[alt])
        cost_swap = abs(tot[r2] - ref[day]) + abs(tot[r1] - ref[alt])
        first, second = (r1, r2) if cost_keep <= cost_swap else (r2, r1)
        pdt[second] = alt
    return pos.map(pdt)


def load():
    d = pd.read_csv(DATA)
    d["state"] = d["States"].replace(ALIAS)
    d["date"] = fix_dates(d["Dates"], d["Usage"], d.groupby("States").cumcount())
    d = d[d["state"].isin(AGRI_STATES)].sort_values(["state", "date"]).reset_index(drop=True)
    return d[["state", "date", "Usage"]].rename(columns={"Usage": "usage"})


def features(d):
    g = d.groupby("state")["usage"]
    d = d.copy()
    d["lag1"], d["lag2"], d["lag7"] = g.shift(1), g.shift(2), g.shift(7)
    d["roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    d["dow"], d["month"] = d["date"].dt.dayofweek, d["date"].dt.month
    doy = d["date"].dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return d.dropna().reset_index(drop=True)


def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p),
            "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100)}


def main():
    d = features(load())
    cut = d["date"].sort_values().unique()[int(d["date"].nunique() * 0.8)]
    tr, te = d[d["date"] < cut], d[d["date"] >= cut]
    Xtr, ytr, Xte, yte = tr[NUM + CAT], tr["usage"], te[NUM + CAT], te["usage"]
    print(f"rows={len(d)} train={len(tr)} test={len(te)} split_date={pd.Timestamp(cut).date()}")

    cv = TimeSeriesSplit(n_splits=3)
    svr_gs = GridSearchCV(Pipeline([("p", prep()), ("m", SVR(kernel="rbf"))]),
                          {"m__C": [10, 100, 300], "m__gamma": ["scale", 0.01], "m__epsilon": [0.1, 1.0]},
                          cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    dt_gs = GridSearchCV(Pipeline([("p", prep()), ("m", DecisionTreeRegressor(random_state=SEED))]),
                         {"m__max_depth": [4, 6, 8, 12, None], "m__min_samples_leaf": [1, 5, 10, 20]},
                         cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    times = {}
    for name, gs in [("SVR", svr_gs), ("Decision Tree", dt_gs)]:
        t = time.time(); gs.fit(Xtr, ytr); times[name] = time.time() - t
        print(name, "best", gs.best_params_)
    svr, dt = svr_gs.best_estimator_, dt_gs.best_estimator_

    voting = VotingRegressor([("svr", svr), ("dt", dt)])
    stack = StackingRegressor([("svr", svr), ("dt", dt)], final_estimator=RidgeCV(),
                              cv=5)
    models = {"SVR": svr, "Decision Tree": dt, "Ensemble (Voting)": voting, "Ensemble (Stacking)": stack}
    for n in ["Ensemble (Voting)", "Ensemble (Stacking)"]:
        t = time.time(); models[n].fit(Xtr, ytr); times[n] = time.time() - t

    preds, res = {}, {}
    for n, m in models.items():
        preds[n] = m.predict(Xte)
        res[n] = metrics(yte.values, preds[n]) | {"train_time_s": round(times[n], 2)}
    # Naive baseline: yesterday's consumption
    res["Naive (lag1)"] = metrics(yte.values, te["lag1"].values) | {"train_time_s": 0.0}
    tab = pd.DataFrame(res).T.round(4)
    print(tab)
    tab.to_csv(OUT / "metrics.csv")

    # Per-state test MAE
    te = te.assign(**{n: p for n, p in preds.items()})
    per = {n: te.groupby("state").apply(lambda g: mean_absolute_error(g["usage"], g[n]), include_groups=False)
           for n in models}
    pd.DataFrame(per).round(3).to_csv(OUT / "per_state_mae.csv")

    for n, m in models.items():
        joblib.dump(m, OUT / (n.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"))
    meta = {"states": AGRI_STATES, "split_date": str(pd.Timestamp(cut).date()), "n_train": len(tr),
            "n_test": len(te), "svr_params": svr_gs.best_params_, "dt_params": dt_gs.best_params_,
            "features": NUM + CAT}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    d.to_csv(OUT / "features.csv", index=False)

    make_figures(d, te, yte, preds, tab, per, dt)


def make_figures(d, te, yte, preds, tab, per, dt):
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    names = list(preds)
    # 1 metric bars
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, m, u in zip(ax, ["MAE", "RMSE", "R2"], ["MU/day", "MU/day", ""]):
        a.bar(range(4), tab.loc[names, m], color=cols)
        a.set_xticks(range(4)); a.set_xticklabels(["SVR", "DT", "Vote", "Stack"])
        a.set_title(f"{m} {('('+u+')') if u else ''}")
        for i, v in enumerate(tab.loc[names, m]): a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Model comparison on held-out test period"); fig.tight_layout(); fig.savefig(FIG / "metrics_bar.png", dpi=150); plt.close(fig)
    # 2 pred vs actual for a state
    s = "Punjab"; m = (te["state"] == s).values
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(te["date"][m], yte[m], "k-", lw=2, label="Actual")
    for n, c in zip(names, cols): ax.plot(te["date"][m], preds[n][m], "--", color=c, label=n, lw=1.2)
    ax.set_title(f"Predicted vs actual daily consumption - {s}"); ax.set_ylabel("Consumption (MU/day)")
    ax.set_xlabel("Date"); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / "pred_vs_actual.png", dpi=150); plt.close(fig)
    # 3 scatter
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
    lim = [yte.min(), yte.max()]
    for a, n, c in zip(ax, names, cols):
        a.scatter(yte, preds[n], s=4, alpha=.4, color=c); a.plot(lim, lim, "k--", lw=1)
        a.set_title(n, fontsize=9); a.set_xlabel("Actual (MU)")
    ax[0].set_ylabel("Predicted (MU)"); fig.tight_layout(); fig.savefig(FIG / "scatter.png", dpi=150); plt.close(fig)
    # 4 residual histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, c in zip(names, cols): ax.hist(yte.values - preds[n], bins=50, alpha=.45, color=c, label=n)
    ax.set_xlabel("Residual (MU/day)"); ax.set_ylabel("Count"); ax.set_title("Residual distribution"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "residuals.png", dpi=150); plt.close(fig)
    # 5 per-state MAE
    p = pd.DataFrame(per)[names]
    ax = p.plot.bar(figsize=(12, 4.5), color=cols, width=.8)
    ax.set_ylabel("MAE (MU/day)"); ax.set_title("Per-state test MAE"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "per_state_mae.png", dpi=150); plt.close()
    # 6 consumption time series of all states
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for s2, g in d.groupby("state"): ax.plot(g["date"], g["usage"], lw=.8, label=s2)
    ax.axvline(pd.Timestamp("2020-03-25"), color="k", ls=":"); ax.text(pd.Timestamp("2020-03-26"), ax.get_ylim()[1]*.95, "Lockdown", fontsize=8)
    ax.set_ylabel("MU/day"); ax.set_title("Daily consumption, agrarian states"); ax.legend(ncol=5, fontsize=6)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG / "timeseries.png", dpi=150); plt.close(fig)
    # 7 monthly seasonality
    mm = d.groupby(["month", "state"])["usage"].mean().unstack()
    ax = mm.plot(figsize=(10, 4.5), legend=False); ax.set_xlabel("Month"); ax.set_ylabel("Mean MU/day"); ax.set_title("Seasonality by month")
    plt.tight_layout(); plt.savefig(FIG / "seasonality.png", dpi=150); plt.close()
    # 8 DT importances
    imp = dt.named_steps["m"].feature_importances_
    fn = dt.named_steps["p"].get_feature_names_out()
    top = pd.Series(imp, fn).sort_values().tail(10)
    ax = top.plot.barh(figsize=(7, 4)); ax.set_title("Decision tree - top feature importances")
    plt.tight_layout(); plt.savefig(FIG / "dt_importance.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
