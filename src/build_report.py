"""Build the technical report (DOCX) from results in models/ and figures/."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

R = Path(__file__).resolve().parents[1]
APP_URL = sys.argv[1] if len(sys.argv) > 1 else "<to be added after deployment>"
REPO_URL = "https://github.com/AnKiT-GaRG2/farm-power-forecast"
met = pd.read_csv(R/"models/metrics.csv", index_col=0)
per = pd.read_csv(R/"models/per_state_mae.csv", index_col=0)
meta = json.loads((R/"models/meta.json").read_text())

# ---- architecture diagram
def arch():
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)
    boxes = [(0.1, "Kaggle CSV\n(state-wise daily\nconsumption)"), (2.3, "Preprocessing\n+ lag/calendar\nfeatures"),
             (4.5, "Train: SVR, DT\n+ Voting/Stacking\n(TimeSeriesSplit)"), (6.7, "Saved models\n(joblib) in\nGitHub repo"),
             (8.9, "Streamlit app on\nStreamlit Community\nCloud (user: browser)")]
    for x, t in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.2), 1.9, 1.3, boxstyle="round,pad=0.05", fc="#dbe9f6", ec="#1f4e79"))
        ax.text(x+.95, 1.85, t, ha="center", va="center", fontsize=8.5)
    for i in range(4):
        x = boxes[i][0]+1.95; ax.annotate("", xy=(boxes[i+1][0]-.02, 1.85), xytext=(x, 1.85), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5.5, 3.1, "System architecture and workflow", ha="center", fontsize=11, weight="bold")
    ax.text(5.5, .6, "GitHub push triggers auto-redeploy on Streamlit Community Cloud", ha="center", fontsize=8.5, style="italic")
    fig.savefig(R/"figures/architecture.png", dpi=150, bbox_inches="tight"); plt.close(fig)
arch()

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
def H(t, l=1): doc.add_heading(t, l)
def P(t): return doc.add_paragraph(t)
def B(items):
    for i in items: doc.add_paragraph(i, style="List Bullet")
def FIG(p, cap, w=6.2):
    doc.add_picture(str(R/"figures"/p), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = P(cap); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.runs[0].italic = True
def TAB(df, fmt="{:.3f}", idx="Model"):
    t = doc.add_table(rows=1, cols=len(df.columns)+1); t.style = "Light Grid Accent 1"
    t.rows[0].cells[0].text = idx
    for j, c in enumerate(df.columns): t.rows[0].cells[j+1].text = str(c)
    for i, r in df.iterrows():
        cells = t.add_row().cells; cells[0].text = str(i)
        for j, v in enumerate(r): cells[j+1].text = fmt.format(v) if isinstance(v, float) else str(v)
    doc.add_paragraph()

# ---- title page
for _ in range(3): doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Ensemble of SVM and Decision Tree Models to Forecast Power Consumption in Indian Farmlands"); r.bold = True; r.font.size = Pt(20)
p = P("Cloud Computing Lab - Project Report"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()
t = doc.add_table(rows=1, cols=2); t.style = "Light Grid Accent 1"
t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Student name", "Roll number"
for n, rl in [("Ankit Garg", "Btech/10218/23"), ("Ayush Mishra", "Btech/10219/23"), ("Anshuman Banerjee", "Btech/10211/23")]:
    c = t.add_row().cells; c[0].text, c[1].text = n, rl
doc.add_paragraph()
P("Course: Cloud Computing Lab (B.Tech)\nSemester: 7th\nSubmission date: 3 November 2026")
P(f"Code repository: {REPO_URL}\nDeployed application: {APP_URL}")
doc.add_page_break()

H("Abstract")
b = met.loc[["SVR","Decision Tree","Ensemble (Voting)","Ensemble (Stacking)"]]
P("Agriculture, mainly electric irrigation pumpsets, is one of the largest electricity-consuming sectors in India, so accurate "
  "short-term demand forecasts help utilities plan generation and avoid shortages during sowing and irrigation seasons. This project "
  "builds a forecasting system for the daily electricity consumption of 15 agriculture-dominated Indian states using a Support Vector "
  "Regressor (SVR), a Decision Tree Regressor (DT) and two ensembles of the two (soft-voting average and stacking with a ridge "
  "meta-learner). Models use lagged consumption, a 7-day rolling mean and calendar features and are evaluated on a chronologically "
  f"held-out period starting {meta['split_date']}. The SVR had the lowest MAE ({b.loc['SVR','MAE']:.2f} MU/day) and MAPE, while the voting ensemble had the lowest RMSE "
  f"({b.loc['Ensemble (Voting)','RMSE']:.2f}) and highest R2 ({b.loc['Ensemble (Voting)','R2']:.3f}) and was better than the single decision tree on MAE, MAPE and RMSE (MAE {b.loc['Ensemble (Voting)','MAE']:.2f} vs {b.loc['Decision Tree','MAE']:.2f}). "
  "The stacking ensemble was the weakest model. All differences between SVR, DT and voting are small relative to the error caused by a regime change (COVID-19 lockdown) inside the test period. "
  "The trained models are served through a Streamlit dashboard deployed on a cloud platform.")

H("1. Introduction")
H("1.1 Background", 2)
P("India is the world's third-largest electricity producer and consumer. Agricultural consumption is dominated by groundwater pumping, which is "
  "strongly seasonal and depends on rainfall, crop cycle and state farm-power policy. Forecasting demand ahead of time is a core task in grid operation.")
H("1.2 Motivation", 2)
P("Single machine-learning models each have weaknesses: SVR is smooth but sensitive to hyper-parameters and scaling, while a decision tree captures "
  "non-linear rules but gives piecewise-constant, high-variance output. Combining heterogeneous learners is a standard way to reduce variance, and "
  "the assignment asks to compare the individual models with their ensemble.")
H("1.3 Problem being addressed", 2)
P("Predict the next-day power consumption (million units, MU) of agrarian states using historical consumption and calendar information, and "
  "quantify whether an SVM + decision-tree ensemble improves over each individual model.")

H("2. Problem Statement and Objectives")
P("Develop an ensemble of SVM and decision tree models to forecast power consumption in Indian farmlands, using a suitable Kaggle dataset, "
  "deploy the model on a cloud platform, and compare individual performance with the ensemble.")
B(["Select and document a suitable public Kaggle dataset and justify its use for agrarian-region forecasting.",
   "Engineer time-series features and train SVR and Decision Tree regressors with time-aware hyper-parameter tuning.",
   "Build voting and stacking ensembles of the two models.",
   "Compare all models with MAE, RMSE, R2, MAPE, training time and a naive baseline.",
   "Build an interactive dashboard and deploy it on a cloud platform."])

H("3. System Architecture and Methodology")
FIG("architecture.png", "Figure 1: Architecture and workflow")
H("3.1 Workflow", 2)
B(["Load the long-format CSV, convert dates (dd/mm/yyyy), rename UP/MP to full state names, keep 15 agrarian states.",
   "Create per-state features: lag-1, lag-2, lag-7, previous-7-day rolling mean, day-of-week, month, sine/cosine of day-of-year, and the state (one-hot).",
   "Chronological 80/20 split by date; hyper-parameters tuned on the training part only with 3-fold TimeSeriesSplit.",
   "Train SVR, Decision Tree, Voting and Stacking ensembles; evaluate on the held-out period; persist with joblib.",
   "Commit models and app to GitHub; Streamlit Community Cloud builds and serves the dashboard."])
H("3.2 Algorithms and formulation", 2)
P("Support Vector Regression (RBF kernel) fits f(x) = sum_i (a_i - a_i*) K(x_i, x) + b with epsilon-insensitive loss, "
  "K(x, x') = exp(-gamma ||x - x'||^2). Hyper-parameters C, gamma and epsilon were tuned by grid search.")
P("Decision Tree Regression recursively splits the feature space to minimise the squared error within leaves; max_depth and min_samples_leaf were tuned.")
P("Voting ensemble: y_hat = (y_svr + y_dt) / 2. Stacking ensemble: y_hat = RidgeCV(y_svr, y_dt) where the meta-learner is trained on 5-fold out-of-fold predictions of the base models.")
P("Metrics: MAE = mean|y - y_hat|; RMSE = sqrt(mean (y - y_hat)^2); R2 = 1 - SSE/SST; MAPE = 100 * mean|(y - y_hat)/y|.")

H("4. Dataset and Experimental Configuration")
TAB(pd.DataFrame({"Value": [
    "Power consumption in India (2019-2020)", "Kaggle, author: twinkle0705 (Twinkle Khanna); licence CC0 Public Domain",
    "https://www.kaggle.com/datasets/twinkle0705/state-wise-power-consumption-in-india",
    "16,599 rows (33 states x 503 days, 2 Jan 2019 - 23 May 2020; five 1-day gaps: 1 Feb-1 Jun 2019); 7,440 rows used (15 states x 496 days after lagging)",
    "Raw: States, Regions, latitude, longitude, Dates, Usage (6). Model inputs: 8 numeric + 1 categorical",
    "Usage - daily electricity consumption in million units (MU)",
    "No missing values; repair of day/month-swapped dates (see below); lag/rolling/calendar features; StandardScaler on numeric, one-hot on state",
    f"Chronological split: train < {meta['split_date']} ({meta['n_train']} rows), test >= {meta['split_date']} ({meta['n_test']} rows)"]},
    index=["Dataset name", "Source", "URL", "Instances", "Features", "Target variable", "Preprocessing", "Train/test methodology"]),
    fmt="{}", idx="Item")
P("Data quality issue and repair: the date column of the Kaggle CSV contains a day/month swap (an Excel locale artefact). 35 rows (true dates 6-12 January-May 2020) are stored as mm/dd and "
  "would parse as June-December 2020, and the 7th of August-December 2019 is stored as mm/dd and collides with 8-12 July 2019. We re-parse dates after 23 May 2020 (the end of the data) as mm/dd and resolve the five colliding "
  "pairs by continuity of the all-state total with the previous day. The pair assignment is a heuristic; its effect is confined to 10 of 7,440 modelled rows (about 0.1%).")
P("Assumption and limitation: the dataset reports total state consumption, not farm-only consumption. Agriculture-dominated states were chosen as a proxy, "
  "because their load is strongly influenced by irrigation pumping; the dataset does not separate agricultural load. This is documented as a limitation.")
FIG("timeseries.png", "Figure 2: Daily consumption of the selected states (dotted line: nationwide lockdown, 25 Mar 2020)")
FIG("seasonality.png", "Figure 3: Monthly mean consumption showing seasonality", 5.2)

H("5. Implementation")
B(["Language/libraries: Python 3.10, pandas, NumPy, scikit-learn 1.7.2 (SVR, DecisionTreeRegressor, VotingRegressor, StackingRegressor, GridSearchCV), matplotlib, Plotly.",
   "Dashboard: Streamlit with three tabs - model comparison, forecast explorer (predicted vs actual per state) and what-if prediction.",
   "Cloud platform: Streamlit Community Cloud (free tier), source from GitHub; Python environment from requirements.txt.",
   "Repository layout: src/train.py (training), src/build_report.py, app.py (dashboard), models/ (trained artefacts), figures/, data/."])
P(f"Best SVR parameters: {meta['svr_params']}. Best Decision Tree parameters: {meta['dt_params']}.")

H("6. Experimental Setup")
B(["Hardware/software: personal laptop, Linux, Python 3.10, scikit-learn 1.7.2.",
   "Dataset/workload: as in Section 4.",
   "Parameters: SVR grid C in {10,100,300}, gamma in {scale,0.01}, epsilon in {0.1,1.0}; DT grid max_depth in {4,6,8,12,None}, min_samples_leaf in {1,5,10,20}; random_state=42.",
   "Number of runs: one chronological train/test split (deterministic); the tuning used 3-fold TimeSeriesSplit.",
   "Metrics: MAE, RMSE, R2, MAPE, training time; naive baseline (yesterday's value)."])

H("7. Results")
H("7.1 Overall comparison", 2)
TAB(met.round(3)); 
FIG("metrics_bar.png", "Figure 4: MAE, RMSE and R2 per model")
H("7.2 Predicted vs actual", 2)
FIG("pred_vs_actual.png", "Figure 5: Predicted vs actual daily consumption for Punjab (test period)")
FIG("scatter.png", "Figure 6: Predicted vs actual scatter plots (dashed line = perfect prediction)")
H("7.3 Errors", 2)
FIG("residuals.png", "Figure 7: Residual distributions", 4.8)
FIG("per_state_mae.png", "Figure 8: MAE per state")
TAB(per.round(2), fmt="{:.2f}", idx="State")
FIG("dt_importance.png", "Figure 9: Decision-tree feature importances", 4.5)
H("7.4 Deployment", 2)
B(["Cloud platform: Streamlit Community Cloud.", "Architecture: GitHub repo -> Streamlit build (pip install requirements.txt) -> app served over HTTPS.",
   "Services/resources: Streamlit Community Cloud container (shared CPU, ~1 GB RAM), GitHub repository.",
   "Procedure: push the repo to GitHub; on share.streamlit.io choose New app, select repository, branch main, main file app.py; click Deploy.",
   f"Application URL: {APP_URL}",
   "Screenshots: see Appendix A (deployment screenshots to be inserted by the students after deploying).",
   "Challenges: pinning scikit-learn to the version used for training (pickled models are version-sensitive); keeping model files small enough for the free tier; cold-start delay when the app has been idle."])

H("8. Analysis and Discussion")
sv, dtm, vo, sk, nv = (met.loc[k] for k in ["SVR","Decision Tree","Ensemble (Voting)","Ensemble (Stacking)","Naive (lag1)"])
P(f"What worked: all learners reach R2 of about 0.88-0.90 because yesterday's consumption is a strong predictor. SVR has the lowest MAE ({sv.MAE:.2f}) and MAPE ({sv['MAPE_%']:.1f}%). "
  f"The voting ensemble has the best RMSE ({vo.RMSE:.2f} vs {sv.RMSE:.2f} for SVR and {dtm.RMSE:.2f} for DT) and best R2 ({vo.R2:.3f}), i.e. it makes the fewest large errors, and it lowers the decision tree's MAE from {dtm.MAE:.2f} to {vo.MAE:.2f} "
  f"({(1-vo.MAE/dtm.MAE)*100:.0f}% lower) and MAPE from {dtm['MAPE_%']:.1f}% to {vo['MAPE_%']:.1f}%. This is the expected variance-reduction effect of averaging two different learners.")
P(f"What did not: the voting ensemble is not better than SVR on MAE ({vo.MAE:.2f} vs {sv.MAE:.2f}), and its RMSE advantage over SVR (about {(1-vo.RMSE/sv.RMSE)*100:.0f}%) and R2 advantage ({vo.R2-sv.R2:.3f}) are small; with one split and no confidence intervals we cannot claim it is statistically significant. "
  f"Stacking was the worst model (MAE {sk.MAE:.2f}, RMSE {sk.RMSE:.2f}), worse even than its two base learners. A likely reason (not tested) is that its meta-learner was fitted on 5-fold out-of-fold predictions that are not time-ordered, so its weights do not transfer to the shifted test period. "
  f"The naive 'yesterday' baseline (MAE {nv.MAE:.2f}, R2 {nv.R2:.3f}) is about as good as the learned models: it beats the decision tree and stacking on MAE and is within 3% of SVR, so the learned models add only modest value over persistence on this dataset.")
P("Why: the test period (from 14 Feb 2020 to 23 May 2020) contains the national COVID-19 lockdown starting 25 March 2020, when consumption changed abruptly. "
  "Models trained on 2019 and early-2020 data have never seen such a regime, so errors are dominated by that structural break (visible in Figure 5 and the long residual tails in Figure 7). "
  "Punjab and Haryana also show sharp step changes (possibly linked to agricultural supply schedules; we did not verify the cause), which a one-day-ahead model can only follow after the change has happened.")
P("Trends: states with large and volatile load (Uttar Pradesh, Punjab, Haryana, Maharashtra) have the largest absolute errors; smaller consumers (Chhattisgarh, Odisha, Bihar, Andhra Pradesh) have the smallest. "
  "The decision tree's feature importance is dominated by lag-1 consumption (Figure 9).")
P("Limitations: (1) only 17 months of data, so a full seasonal cycle is never repeated and the test set is atypical; (2) state totals rather than farm-level data; "
  "(3) no weather, rainfall, crop or tariff covariates; (4) a single split and no confidence intervals, so differences of a few percent between models should not be over-interpreted; "
  "(5) date repair heuristic (Section 4); (6) five missing days in 2019 make lag-1/lag-7 features span two days at those points; (7) the ensemble result is specific to this dataset and period.")

H("9. Conclusion")
P("An SVR, a decision tree and two SVR+DT ensembles were built and compared for forecasting daily power consumption of 15 agrarian Indian states. "
  f"SVR achieved the lowest MAE ({sv.MAE:.2f} MU/day) and MAPE; the voting ensemble achieved the lowest RMSE ({vo.RMSE:.2f}) and highest R2 ({vo.R2:.3f}) and improved clearly on the decision tree, but the margin over SVR is small. "
  "Stacking did not help. The work was deployed as an interactive Streamlit dashboard on a cloud platform. The main findings are that ensembling gives a modest, not dramatic, gain over the best individual model here, "
  "and that regime changes such as the lockdown, together with data-quality issues in the public dataset, limit all models.")
H("10. Future Scope")
B(["Use farm-level or feeder-level agricultural load data (for example state discom pumpset consumption) and multi-year data.",
   "Add weather, rainfall, groundwater and crop-calendar features.", "Weighted ensembles or gradient boosting / LSTM comparison; walk-forward evaluation.",
   "Multi-day-ahead forecasting with prediction intervals; automated retraining pipeline on the cloud."])

H("11. References")
B(["Khanna, T. (2020). Power consumption in India (2019-2020). Kaggle. https://www.kaggle.com/datasets/twinkle0705/state-wise-power-consumption-in-india (CC0).",
   "Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.",
   "Drucker et al. (1997). Support Vector Regression Machines. NIPS 9.", "Breiman et al. (1984). Classification and Regression Trees. Wadsworth.",
   "Wolpert, D. (1992). Stacked Generalization. Neural Networks 5(2).", "Streamlit documentation: https://docs.streamlit.io; Streamlit Community Cloud: https://streamlit.io/cloud",
   "pandas, NumPy, matplotlib and Plotly documentation."])
H("Appendix A: Deployment screenshots")
P("[Insert screenshots of the successful Streamlit Community Cloud deployment and the running dashboard here.]")
doc.save(R/"Project_Report.docx"); print("saved")
