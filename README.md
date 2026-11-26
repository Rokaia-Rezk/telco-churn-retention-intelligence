# Telco Churn Retention Intelligence

End-to-end customer churn project on the IBM Telco dataset: a leak-free pipeline, tuned Gradient Boosting, explainable AI (SHAP and permutation importance), K-Means customer segments with retention offers, and a Streamlit dashboard for retention teams.

Most churn projects stop at "this customer will leave." This one goes further: it explains **why** each customer is at risk, groups customers into **segments**, and recommends a **retention offer** for each one, all in an app a retention manager can use directly.

**Live demo:**(https://telco-churn-retention-intelligence-k8mucssrfsqjbdac7rfty8.streamlit.app/)


---

## Business Problem

Keeping a customer is cheaper than winning a new one. A retention team has a limited budget, so it needs to know:

1. **Who** is likely to leave (and who to call first).
2. **Why** they are likely to leave.
3. **What** offer to make to each group.

## Dataset

| Item | Value |
|---|---|
| Source | IBM Telco Customer Churn sample data (fictional company) |
| Customers | 7,043 |
| Features used | 19 (demographics, services, contract, billing) |
| Target | `Churn` (1,869 customers left, **26.5%**) |
| Split | 80/20 stratified, `random_state=42` (5,634 train / 1,409 test) |

Because the target is imbalanced, a model that always predicts "stay" already gets 73.5% accuracy while catching nobody. The project therefore uses **PR-AUC** and **recall** as the main metrics, not accuracy.

## What's New Compared With a Basic Prediction Project

| Area | What was done |
|---|---|
| Leak-free preprocessing | Scaling and one-hot encoding live inside one scikit-learn `Pipeline`, so nothing leaks from validation folds |
| Evaluation | Stratified 5-fold cross-validation, PR-AUC as the main metric |
| Model selection | Four models compared and tuned with `GridSearchCV`; the winner was chosen by a rule fixed in advance |
| Threshold | Chosen from **out-of-fold** predictions on the training set only (best F1), not guessed |
| Test set | Used once, at the end |
| Explainability | SHAP and permutation importance, checked against the exploratory analysis |
| Segmentation | K-Means (k = 3) fitted without the target, then profiled, each segment with a retention offer |
| Deployment | Streamlit dashboard with a priority list, single-customer scoring, and segment offers |

## Workflow

```
Data understanding -> Quality checks -> Cleaning -> Train/test split
-> EDA (train only) -> Leak-free pipeline -> Cross-validation of 4 models
-> Tuning -> Threshold selection -> Test evaluation -> XAI (SHAP + permutation)
-> K-Means segments -> Retention offers -> Streamlit dashboard
```

## Key Findings (exploratory analysis, training set)

- **Tenure:** churn is 52.2% in the first 6 months and falls to 9.9% for customers with 49 to 72 months.
- **Contract:** month-to-month customers churn at 42.7%, versus 11.1% (one-year) and 2.9% (two-year).
- **Internet service:** Fiber optic customers churn at 42.1%, versus 18.7% for DSL.
- **Payment method:** electronic check users churn at 45.7%, versus 14.9% for credit card (automatic).
- **Protection services:** among Fiber customers, churn is 49.6% without OnlineSecurity and 21.9% with it.
- **Price alone is not the cause:** churners pay more on average (74.9 vs 61.3 per month), but inside each internet plan they pay the same or less. The gap comes from the larger share of Fiber customers among churners.
- **Gender has no effect** (26.6% vs 26.4%).
- **Revenue at risk:** churners are 26.5% of customers but 30.6% of monthly revenue.

## Model

**Final model:** Gradient Boosting (`n_estimators=200`, `learning_rate=0.05`, `max_depth=2`), chosen by the highest cross-validated PR-AUC.

| Model | CV PR-AUC | Std |
|---|---|---|
| **Gradient Boosting (chosen)** | **0.667** | 0.021 |
| Random Forest | 0.665 | 0.019 |
| Logistic Regression | 0.660 | 0.020 |
| KNN | 0.633 | 0.019 |

After tuning, the top models are statistically tied (the gap between the top two is 0.0013, smaller than one standard deviation), so the choice follows the rule fixed in advance.

**Test-set results** (1,409 customers, evaluated once, threshold = 0.332):

| Metric | Model | Baseline ("always stay") |
|---|---|---|
| Recall | **0.741** | 0.000 |
| Precision | 0.560 | 0.000 |
| F1 | 0.638 | 0.000 |
| PR-AUC | **0.662** | 0.265 |
| ROC-AUC | 0.845 | 0.500 |
| Accuracy | 0.776 | 0.735 |

The tuned threshold raises recall from about 0.51 (at the default 0.5) to 0.74: the model catches roughly three out of four churners, at the cost of sending some unnecessary offers to loyal customers. This assumes a missed churner costs more than an unnecessary offer.

## Explainable AI

Two methods were applied to the final model, so the conclusion does not depend on a single technique.

| Rank | Permutation importance | SHAP |
|---|---|---|
| 1 | tenure | Contract |
| 2 | Contract | tenure |
| 3 | InternetService | InternetService |

- The two methods agree on the top three features: **Contract, tenure, and InternetService** (Spearman rank correlation 0.79).
- Direction of effect: month-to-month contract, Fiber optic, electronic check, paperless billing, and high monthly charges **increase** risk; long tenure, two-year contract, no internet service, and TechSupport / OnlineSecurity **decrease** it.
- The model agrees with the exploratory analysis in 14 of 21 compared columns, and none of the disagreements is among the 15 most influential columns.

> XAI explains what the model relies on. It does not prove that a feature causes churn.

## Customer Segments and Retention Offers

K-Means grouped all 7,043 customers by profile **without seeing churn**. Churn was used afterwards only to describe the groups.

| Segment | Customers | Churn rate | Profile | Retention offer |
|---|---|---|---|---|
| **At-risk: new, month-to-month** | 2,987 (42%) | 46.6% | 14.5 months average tenure, 90% month-to-month, 58% Fiber, 54% electronic check | Discount for a 1-year or 2-year contract; free 3-month TechSupport / OnlineSecurity bundle for Fiber customers; small bill credit for automatic payment; onboarding programme and check-in call in the first 6 months |
| **Loyal: long tenure, high spend** | 2,214 (31%) | 14.3% | 58 months average tenure, highest monthly charge, 43% on a two-year contract | Low risk: loyalty rewards and upsell, no retention discount needed |
| **Phone-only, low spend** | 1,842 (26%) | 8.7% | 83% have no internet service, lowest monthly charge | Low risk: loyalty rewards and upsell, no retention discount needed |

The first segment is 42% of customers but holds about **74% of all churners**.

The silhouette score for k = 3 is about 0.20, so these segments are a practical grouping, not sharply separated clusters. The offers are rules tied to the EDA and XAI findings; their real effect must be measured with a pilot that keeps a control group.

## Dashboard

| Page | What it shows |
|---|---|
| Overview | KPIs and churn by tenure, contract, internet service, and payment method |
| Priority customers | Active customers ranked by churn risk, with filters, risk drivers, suggested offer, and CSV download |
| Score a customer | Enter or edit one customer's details and get the risk, the main reasons, the segment, and the offer |
| Why customers leave | Permutation and SHAP importance, direction of effects, and evidence from the data |
| Segments and offers | The three segments, their profile, and their offers |
| Model and method | Metrics, confusion matrix, model comparison, and limits |

Scores for customers in the training part of the data are **out-of-fold** predictions (the model never saw that customer when scoring it); the rest are held-out test predictions.

## Project Structure

```
telco-churn-retention-intelligence/
├── .streamlit/              # Streamlit theme / configuration
├── telco_churn_app/
│   ├── app.py               # Streamlit pages
│   ├── core.py              # Cleaning, pipeline, model, threshold, segments, explanations
│   ├── charts.py            # Matplotlib charts in the project palette
│   └── data/
│       └── telco_churn.csv  # IBM Telco dataset
├── notebooks/
│   └── Telco_Churn_Full_Project_Done.ipynb   # Full analysis (EDA, models, XAI, K-Means)
├── assets/
│   └── dashboard.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

```bash
git clone https://github.com/Rokaia-Rezk/telco-churn-retention-intelligence.git
cd telco-churn-retention-intelligence
python -m venv venv
venv\Scripts\activate        # Windows (on Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run telco_churn_app/app.py
```

The model is trained on the first load (about 10 seconds) and then cached.

## Limitations

- The data is a public sample from a fictional company.
- XAI explains the model but does not prove cause.
- The threshold assumes a missed churner costs more than an unnecessary offer.
- Lost revenue is not profit: the data has no cost information.
- Offers are rule-based; a pilot with a control group is needed to measure their real effect.

## Tech Stack

Python, pandas, NumPy, scikit-learn, SHAP, Matplotlib, Streamlit

## Team

Rokaia Rezk, Rofaida Essam, Yomna Waleed, Roqia El-Gendy, Ganna Essa

[LinkedIn](https://www.linkedin.com/in/rokaia-rezk-0761052bb/) | [GitHub](https://github.com/Rokaia-Rezk)
