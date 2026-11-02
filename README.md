# Power Consumption Forecasting for Indian Farmland States (SVR + Decision Tree Ensemble)

Cloud Computing Lab project. Forecasts next-day electricity consumption (MU) of 15 agriculture-dominated Indian states with
an SVR, a Decision Tree and two ensembles (soft voting, stacking), and serves the results in a Streamlit dashboard.

**Team:** Ankit Garg (Btech/10218/23), Ayush Mishra (Btech/10219/23), Anshuman Banerjee (Btech/10211/23)

**Dataset:** [Power consumption in India (2019-2020)](https://www.kaggle.com/datasets/twinkle0705/state-wise-power-consumption-in-india)
by twinkle0705 on Kaggle (CC0). 33 states x 503 days. The source CSV has day/month-swapped dates; `src/train.py::fix_dates` repairs them (see report, Section 4).

## Results (held-out period 2020-02-14 to 2020-05-23, includes COVID lockdown)

| Model | MAE | RMSE | R2 | MAPE % |
|---|---|---|---|---|
| SVR | 16.88 | 33.75 | 0.897 | 9.27 |
| Decision Tree | 18.95 | 33.17 | 0.901 | 10.50 |
| Ensemble (Voting) | 17.45 | 32.97 | 0.902 | 9.64 |
| Ensemble (Stacking) | 19.57 | 36.00 | 0.883 | 10.80 |
| Naive (yesterday) | 17.30 | 34.55 | 0.892 | 9.52 |

SVR is best on MAE/MAPE, the voting ensemble on RMSE/R2; the gaps are small. See `Project_Report.pdf` for the full analysis.

## Run
```bash
pip install -r requirements.txt
python src/train.py          # trains, writes models/ and figures/
streamlit run app.py         # dashboard
python src/build_report.py   # regenerates Project_Report.docx (needs python-docx)
```

## Live app
https://farm-power-forecast-ankit.streamlit.app/

## Cloud deployment (Streamlit Community Cloud)
1. Push this repo to GitHub. 2. Go to https://share.streamlit.io -> New app. 3. Select the repo, branch `main`, file `app.py`. 4. Deploy.
The pinned `scikit-learn==1.7.2` in `requirements.txt` matches the version used to pickle the models.
