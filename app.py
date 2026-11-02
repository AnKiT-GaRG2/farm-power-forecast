"""Streamlit dashboard: forecast power consumption of agrarian Indian states."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
M = ROOT / "models"
st.set_page_config(page_title="Farm Power Forecast", layout="wide")

FILES = {"SVR": "svr", "Decision Tree": "decision_tree",
         "Ensemble (Voting)": "ensemble_voting", "Ensemble (Stacking)": "ensemble_stacking"}


@st.cache_resource
def load_models():
    return {n: joblib.load(M / f"{f}.joblib") for n, f in FILES.items()}


@st.cache_data
def load_data():
    d = pd.read_csv(M / "features.csv", parse_dates=["date"])
    return d, pd.read_csv(M / "metrics.csv", index_col=0), pd.read_csv(M / "per_state_mae.csv", index_col=0), json.loads((M / "meta.json").read_text())


models = load_models()
d, metrics, per_state, meta = load_data()
FEATS = meta["features"]
split = pd.Timestamp(meta["split_date"])

st.title("Power Consumption Forecasting - Indian Agrarian States")
st.caption("SVR + Decision Tree ensemble | Data: Kaggle 'Power consumption in India (2019-2020)' by twinkle0705 (CC0)")

tab1, tab2, tab3 = st.tabs(["Model comparison", "Forecast explorer", "What-if prediction"])

with tab1:
    st.subheader("Held-out test performance (from %s)" % split.date())
    st.dataframe(metrics.style.format("{:.3f}").highlight_min(subset=["MAE", "RMSE", "MAPE_%"], color="#c6efce")
                 .highlight_max(subset=["R2"], color="#c6efce"), width="stretch")
    c1, c2 = st.columns(2)
    long = metrics.reset_index().rename(columns={"index": "Model"}).melt("Model", ["MAE", "RMSE"])
    c1.plotly_chart(px.bar(long, x="Model", y="value", color="variable", barmode="group",
                           labels={"value": "Error (MU/day)", "variable": "Metric"}, title="MAE and RMSE by model"), width="stretch")
    ps = per_state.reset_index().rename(columns={per_state.index.name or "state": "State"}).melt("State", var_name="Model", value_name="MAE")
    c2.plotly_chart(px.bar(ps, x="State", y="MAE", color="Model", barmode="group",
                           labels={"MAE": "MAE (MU/day)"}, title="Per-state MAE"), width="stretch")

with tab2:
    c1, c2 = st.columns(2)
    state = c1.selectbox("State", meta["states"])
    sel = c2.multiselect("Models", list(models), default=["SVR", "Decision Tree", "Ensemble (Voting)"])
    s = d[(d["state"] == state) & (d["date"] >= split)].copy()
    fig = go.Figure(go.Scatter(x=s["date"], y=s["usage"], name="Actual", line=dict(color="black", width=3)))
    for n in sel:
        fig.add_trace(go.Scatter(x=s["date"], y=models[n].predict(s[FEATS]), name=n, line=dict(dash="dash")))
    fig.update_layout(title=f"Predicted vs actual - {state}", xaxis_title="Date", yaxis_title="Consumption (MU/day)")
    st.plotly_chart(fig, width="stretch")
    full = d[d["state"] == state]
    st.plotly_chart(px.line(full, x="date", y="usage", labels={"usage": "MU/day", "date": "Date"},
                            title=f"Full history - {state}"), width="stretch")

with tab3:
    st.write("Enter recent consumption to predict the next day for a state.")
    c = st.columns(4)
    state = c[0].selectbox("State ", meta["states"], key="wi")
    l1 = c[1].number_input("Yesterday (MU)", 0.0, 1000.0, float(d[d.state == state]["usage"].iloc[-1]))
    l2 = c[2].number_input("2 days ago (MU)", 0.0, 1000.0, l1)
    l7 = c[3].number_input("7 days ago (MU)", 0.0, 1000.0, l1)
    r7 = st.number_input("Mean of previous 7 days (MU)", 0.0, 1000.0, l1)
    dt = st.date_input("Target date", pd.Timestamp("2020-06-01"))
    t = pd.Timestamp(dt); doy = t.dayofyear
    X = pd.DataFrame([{"lag1": l1, "lag2": l2, "lag7": l7, "roll7": r7, "dow": t.dayofweek, "month": t.month,
                       "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25), "state": state}])
    out = pd.DataFrame({"Model": list(models), "Forecast (MU)": [float(m.predict(X)[0]) for m in models.values()]})
    st.dataframe(out.style.format({"Forecast (MU)": "{:.2f}"}), width="stretch")
    st.plotly_chart(px.bar(out, x="Model", y="Forecast (MU)", title="Next-day forecast by model"), width="stretch")
