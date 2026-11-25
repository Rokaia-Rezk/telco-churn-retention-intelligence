# telco-churn-retention-intelligence
End-to-end customer churn project on the IBM Telco dataset: leak-free pipeline, tuned Gradient Boosting, explainable AI (SHAP and permutation importance), K-Means customer segments with retention offers, and a Streamlit dashboard for retention teams.
Most churn projects stop at "this customer will leave." This one goes further: it explains why each customer is at risk, groups customers into segments, and recommends a retention offer for each one, all in an app a retention manager can use directly.

What's new compared with a basic prediction project:

Leak-free preprocessing and cross-validation, with PR-AUC as the main metric because the data is imbalanced.
Four models compared, then Gradient Boosting tuned, with the decision threshold chosen from out-of-fold predictions.
Explainable AI (SHAP and permutation importance) to rank the factors that drive churn.
K-Means segmentation with an offer for each risky segment.
A deployed Streamlit dashboard: priority customer list, single-customer scoring, and segment offers.
