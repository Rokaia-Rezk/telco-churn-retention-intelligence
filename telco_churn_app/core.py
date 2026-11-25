"""Core logic of the Telco Churn Retention app (no Streamlit code here, so it can be tested alone).

Everything mirrors the project notebook: same cleaning, same 80/20 stratified split (random_state=42),
same preprocessing Pipeline, same final model (Gradient Boosting with the tuned parameters),
same threshold rule (best F1 on out-of-fold predictions of the training set) and the same K-Means (k=3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "telco_churn.csv"

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
PASSTHROUGH_COLS = ["SeniorCitizen"]
CAT_COLS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUM_COLS + PASSTHROUGH_COLS + CAT_COLS      # the 19 model inputs

SEGMENT_ORDER = ["At-risk: new, month-to-month", "Loyal: long tenure, high spend", "Phone-only, low spend"]

# Static results copied from the project notebook run (Gradient Boosting, test set, 20 repeats)
PERMUTATION_TOP = {"tenure": 0.1467, "Contract": 0.0856, "InternetService": 0.0787, "MonthlyCharges": 0.0204,
                   "TechSupport": 0.0155, "PhoneService": 0.0074, "OnlineSecurity": 0.0074,
                   "TotalCharges": 0.0069, "PaperlessBilling": 0.0067, "PaymentMethod": 0.0055}
SHAP_TOP = {"Contract": 0.6203, "tenure": 0.4727, "InternetService": 0.4333, "PaymentMethod": 0.1622,
            "PaperlessBilling": 0.1262, "MonthlyCharges": 0.1187, "TechSupport": 0.1024,
            "OnlineSecurity": 0.0907, "MultipleLines": 0.0855, "TotalCharges": 0.0687}


# ------------------------------------------------------------------ data
def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Same cleaning as Section 3 of the notebook. customerID is kept as a column for display."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    for col in CAT_COLS + ["customerID"]:
        df[col] = df[col].str.strip()
    df[CAT_COLS] = df[CAT_COLS].replace({"No internet service": "No", "No phone service": "No"})
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("pass", "passthrough", PASSTHROUGH_COLS),
    ])


def make_model() -> Pipeline:
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42)
    return Pipeline([("prep", make_preprocessor()), ("model", gb)])


# ------------------------------------------------------------------ rules (from the EDA / XAI findings)
def risk_drivers(row: pd.Series) -> list[str]:
    """Plain-language risk drivers present in a customer's profile (EDA + SHAP findings)."""
    out = []
    if row["Contract"] == "Month-to-month":
        out.append("Month-to-month contract")
    if row["tenure"] <= 6:
        out.append(f"New customer ({int(row['tenure'])} months)")
    if row["InternetService"] == "Fiber optic":
        out.append("Fiber optic")
        if row["TechSupport"] == "No" or row["OnlineSecurity"] == "No":
            out.append("Fiber without TechSupport / OnlineSecurity")
    if row["PaymentMethod"] == "Electronic check":
        out.append("Electronic check")
    return out


def suggest_offers(profile_row: pd.Series, overall: dict) -> list[str]:
    """Same rules as Section 14.5 of the notebook."""
    if profile_row["lift"] <= 1:
        return ["Low risk: loyalty rewards and upsell, no retention discount needed"]
    offers = []
    if profile_row["pct_month_to_month"] > overall["pct_month_to_month"] + 0.10:
        offers.append("Discount for moving to a 1-year or 2-year contract")
    if profile_row["pct_fiber"] > overall["pct_fiber"] + 0.10 and profile_row["pct_tech_support"] < 0.5:
        offers.append("Free 3-month TechSupport / OnlineSecurity bundle for Fiber customers")
    if profile_row["pct_electronic_check"] > overall["pct_electronic_check"] + 0.10:
        offers.append("Small bill credit for switching to automatic payment")
    if profile_row["avg_tenure"] < overall["avg_tenure"] * 0.6:
        offers.append("Onboarding programme and a check-in call during the first 6 months")
    return offers or ["Retention follow-up call"]


# ------------------------------------------------------------------ training everything once
def build_artifacts(path: Path = DATA_PATH) -> dict:
    df = load_clean(path)
    X = df[FEATURES]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = make_model()
    model.fit(X_train, y_train)

    # threshold = best F1 on out-of-fold predictions of the training set (same rule as the notebook)
    oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1c = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[f1c[:-1].argmax()])

    test_proba = model.predict_proba(X_test)[:, 1]
    pred = (test_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, test_proba), "pr_auc": average_precision_score(y_test, test_proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "baseline_accuracy": float(1 - y_test.mean()), "baseline_pr_auc": float(y_test.mean()),
        "oof_precision": precision_score(y_train, (oof >= threshold).astype(int)),
        "oof_recall": recall_score(y_train, (oof >= threshold).astype(int)),
    }

    # honest risk score for every customer: out-of-fold for training rows, model output for test rows
    risk = pd.Series(index=df.index, dtype=float)
    risk.loc[X_train.index] = oof
    risk.loc[X_test.index] = test_proba

    # K-Means segments (k = 3), fitted without Churn, exactly as in Section 14
    seg_prep = make_preprocessor()
    Z = seg_prep.fit_transform(X)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
    labels = pd.Series(kmeans.labels_, index=df.index)

    prof = X.copy()
    prof["Churn"] = y
    prof["segment"] = labels
    g = prof.groupby("segment")
    profile = pd.DataFrame({
        "customers": g.size(), "churn_rate": g["Churn"].mean(), "avg_tenure": g["tenure"].mean(),
        "avg_monthly": g["MonthlyCharges"].mean(), "pct_senior": g["SeniorCitizen"].mean(),
        "pct_month_to_month": g["Contract"].apply(lambda s: (s == "Month-to-month").mean()),
        "pct_two_year": g["Contract"].apply(lambda s: (s == "Two year").mean()),
        "pct_fiber": g["InternetService"].apply(lambda s: (s == "Fiber optic").mean()),
        "pct_no_internet": g["InternetService"].apply(lambda s: (s == "No").mean()),
        "pct_tech_support": g["TechSupport"].apply(lambda s: (s == "Yes").mean()),
        "pct_electronic_check": g["PaymentMethod"].apply(lambda s: (s == "Electronic check").mean()),
    })
    profile["share"] = profile["customers"] / len(prof)
    profile["lift"] = profile["churn_rate"] / y.mean()
    profile["lost_customers"] = (profile["customers"] * profile["churn_rate"]).round(0)
    order = profile.sort_values("churn_rate", ascending=False).index.tolist()
    names = {seg: SEGMENT_ORDER[i] for i, seg in enumerate(order)}      # 3 segments: highest, middle, lowest churn
    profile["name"] = profile.index.map(names)
    overall = {
        "pct_month_to_month": float((X["Contract"] == "Month-to-month").mean()),
        "pct_electronic_check": float((X["PaymentMethod"] == "Electronic check").mean()),
        "pct_fiber": float((X["InternetService"] == "Fiber optic").mean()),
        "avg_tenure": float(X["tenure"].mean()),
    }
    profile["offers"] = [suggest_offers(profile.loc[s], overall) for s in profile.index]
    profile = profile.loc[order]

    customers = df.copy()
    customers["risk"] = risk
    customers["segment"] = labels.map(names)
    customers["flagged"] = customers["risk"] >= threshold
    customers["drivers"] = customers.apply(risk_drivers, axis=1)

    rng = np.random.RandomState(42)
    reference = X_train.iloc[rng.choice(len(X_train), size=150, replace=False)].reset_index(drop=True)

    return {"df": df, "customers": customers, "model": model, "threshold": threshold, "metrics": metrics,
            "profile": profile, "names": names, "seg_prep": seg_prep, "kmeans": kmeans,
            "reference": reference, "n_train": len(X_train), "n_test": len(X_test), "overall_churn": float(y.mean())}


# ------------------------------------------------------------------ scoring and explanations
def score_customer(art: dict, row: pd.DataFrame) -> dict:
    prob = float(art["model"].predict_proba(row[FEATURES])[0, 1])
    seg = int(art["kmeans"].predict(art["seg_prep"].transform(row[FEATURES]))[0])
    name = art["names"][seg]
    return {"prob": prob, "flagged": prob >= art["threshold"], "segment": name,
            "offers": art["profile"].loc[seg, "offers"] if seg in art["profile"].index else []}


def explain_customer(art: dict, row: pd.DataFrame) -> pd.DataFrame:
    """Contribution of each original feature: p(customer) minus the average p when that feature is replaced
    by values of typical customers (150 random training rows). Positive = pushes the churn risk up."""
    row = row[FEATURES].reset_index(drop=True)
    ref = art["reference"]
    base = float(art["model"].predict_proba(row)[0, 1])
    blocks = []
    for f in FEATURES:
        tmp = pd.concat([row] * len(ref), ignore_index=True)
        tmp[f] = ref[f].values
        blocks.append(tmp)
    big = pd.concat(blocks, ignore_index=True)
    p = art["model"].predict_proba(big)[:, 1].reshape(len(FEATURES), len(ref)).mean(axis=1)
    out = pd.DataFrame({"feature": FEATURES, "value": [row.loc[0, f] for f in FEATURES], "contribution": base - p})
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def churn_by(df: pd.DataFrame, col: str, bins=None, labels=None) -> pd.Series:
    s = df[col] if bins is None else pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df.groupby(s, observed=True)["Churn"].mean() * 100
