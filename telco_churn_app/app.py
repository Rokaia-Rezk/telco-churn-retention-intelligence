"""Telco Churn Retention Dashboard (Streamlit).

Run locally:   streamlit run app.py
"""
import pandas as pd
import streamlit as st

import charts
import core

AUTHOR = "Rokaia Rezk"
LINKEDIN = "https://www.linkedin.com/in/rokaia-rezk-0761052bb/"
GITHUB = "https://github.com/Rokaia-Rezk"
NAVY, ORANGE, CREAM, GREY = "#14213D", "#E07B1F", "#F6F5F0", "#5B6472"

st.set_page_config(page_title="Telco Churn Retention", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1250px; }}
  h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
  #MainMenu {{ visibility: hidden; }} footer {{ visibility: hidden; }}
  .kpi {{ background: #fff; border: 1px solid #DAD7CB; border-radius: 10px; padding: 16px 18px; }}
  .kpi .v {{ font-size: 2rem; font-weight: 700; color: {ORANGE}; line-height: 1.1; }}
  .kpi .l {{ font-size: 0.82rem; color: {GREY}; margin-top: 4px; }}
  .card {{ background: #fff; border: 1px solid #DAD7CB; border-radius: 10px; padding: 18px 20px; height: 100%; }}
  .card h4 {{ margin: 0 0 6px 0; color: {NAVY}; }}
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }}
  .tag-high {{ background: {ORANGE}; color: #fff; }}
  .tag-low {{ background: #3B6EA8; color: #fff; }}
  .eyebrow {{ color: {ORANGE}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }}
  .muted {{ color: {GREY}; font-size: 0.88rem; }}
  .side-brand {{ font-size: 1.35rem; font-weight: 700; line-height: 1.15; margin-bottom: 2px; }}
  .side-link {{ display: block; text-align: center; padding: 8px 10px; margin-top: 8px; border: 1px solid rgba(255,255,255,0.35);
               border-radius: 8px; color: #fff !important; text-decoration: none !important; font-size: 0.9rem; }}
  .side-link:hover {{ background: {ORANGE}; border-color: {ORANGE}; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the model on first load (about 10 seconds)...")
def get_artifacts():
    return core.build_artifacts()


art = get_artifacts()
cust = art["customers"]
thr = art["threshold"]
profile = art["profile"]


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def header(eyebrow, title, text=None):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if text:
        st.markdown(f'<p class="muted">{text}</p>', unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
PAGES = ["Overview", "Priority customers", "Score a customer", "Why customers leave", "Segments and offers", "Model and method"]
with st.sidebar:
    st.markdown('<div class="eyebrow">Customer retention</div><div class="side-brand">Telco Churn<br>Dashboard</div>', unsafe_allow_html=True)
    st.write("")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.write("")
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.78rem;opacity:0.75;">Built by</div>'
        f'<div style="font-size:1.1rem;font-weight:700;">{AUTHOR}</div>'
        f'<a class="side-link" href="{LINKEDIN}" target="_blank" rel="noopener noreferrer">LinkedIn</a>'
        f'<a class="side-link" href="{GITHUB}" target="_blank" rel="noopener noreferrer">GitHub</a>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="font-size:0.72rem;opacity:0.6;margin-top:14px;">Demo on the public IBM Telco sample data (fictional company).</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ pages
def page_overview():
    header("Overview", "Who is leaving, and how much is at stake",
           "7,043 customers of a fictional telecom company. Churn means the customer left the company.")
    active = cust[cust["Churn"] == 0]
    flagged = active[active["flagged"]]
    lost_rev = cust.loc[cust["Churn"] == 1, "MonthlyCharges"].sum() / cust["MonthlyCharges"].sum() * 100
    c = st.columns(4)
    kpi(c[0], f"{len(cust):,}", "customers in the data")
    kpi(c[1], f"{art['overall_churn'] * 100:.1f}%", "churn rate (1,869 customers left)")
    kpi(c[2], f"{lost_rev:.1f}%", "of monthly revenue was with customers who left")
    kpi(c[3], f"{len(flagged):,}", "active customers flagged for retention")
    st.write("")
    left, right = st.columns(2)
    tb = core.churn_by(cust, "tenure", bins=[0, 6, 12, 24, 48, 72], labels=["0-6 mo", "7-12", "13-24", "25-48", "49-72"])
    left.pyplot(charts.churn_bars(tb, "Churn by months with the company", art["overall_churn"] * 100))
    right.pyplot(charts.churn_bars(core.churn_by(cust, "Contract"), "Churn by contract type", art["overall_churn"] * 100))
    left, right = st.columns(2)
    left.pyplot(charts.churn_bars(core.churn_by(cust, "InternetService"), "Churn by internet service", art["overall_churn"] * 100))
    right.pyplot(charts.churn_bars(core.churn_by(cust, "PaymentMethod"), "Churn by payment method", art["overall_churn"] * 100, rotate=15))
    st.caption("Charts use all 7,043 customers; the presentation's EDA used the training split only, so figures differ by a few tenths of a point.")
    st.info("Reading: new customers on month-to-month contracts with Fiber optic and electronic check leave the most. "
            "Open **Priority customers** to see who to call first.")


def page_priority():
    header("Priority customers", "Who should the retention team call first",
           "Customers are ranked by predicted churn risk. Active customers are shown by default, since customers who already left cannot be retained.")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
    segs = f1.multiselect("Segment", sorted(cust["segment"].unique()), default=sorted(cust["segment"].unique()))
    contracts = f2.multiselect("Contract", sorted(cust["Contract"].unique()), default=sorted(cust["Contract"].unique()))
    min_risk = f3.slider("Minimum churn risk", 0.0, 1.0, float(round(thr, 2)), 0.01)
    include_left = f4.checkbox("Include customers who already left", value=False)

    d = cust[cust["segment"].isin(segs) & cust["Contract"].isin(contracts) & (cust["risk"] >= min_risk)]
    if not include_left:
        d = d[d["Churn"] == 0]
    d = d.sort_values("risk", ascending=False)

    c = st.columns(3)
    kpi(c[0], f"{len(d):,}", "customers match the filters")
    kpi(c[1], f"{d['risk'].mean() * 100:.0f}%" if len(d) else "-", "average churn risk")
    kpi(c[2], f"{d['MonthlyCharges'].sum():,.0f}" if len(d) else "-", "monthly charges of these customers")
    st.write("")

    offer_by_segment = {row["name"]: " | ".join(row["offers"]) for _, row in profile.iterrows()}
    table = pd.DataFrame({
        "Customer ID": d["customerID"],
        "Churn risk": d["risk"] * 100,
        "Segment": d["segment"],
        "Contract": d["Contract"],
        "Tenure (months)": d["tenure"],
        "Monthly charges": d["MonthlyCharges"],
        "Risk drivers present": d["drivers"].apply(lambda x: ", ".join(x) if x else "none of the main drivers"),
        "Suggested offer": d["segment"].map(offer_by_segment),
    })
    st.dataframe(
        table.head(300),
        hide_index=True,
        column_config={
            "Churn risk": st.column_config.ProgressColumn("Churn risk", format="%.0f%%", min_value=0, max_value=100),
            "Monthly charges": st.column_config.NumberColumn("Monthly charges", format="%.2f"),
        },
    )
    st.download_button("Download the full filtered list (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="priority_customers.csv", mime="text/csv")
    st.caption(f"Showing the first 300 rows. Flag threshold: {thr:.3f}. Scores for customers in the training part of the data are "
               "out-of-fold predictions (the model never saw that customer when scoring it); the rest are held-out test predictions.")


def page_score():
    header("Score a customer", "Check one customer's churn risk",
           "Start from an existing customer or from a typical profile, change any value, and see the risk, the reasons and the offer.")
    ids = ["Typical customer"] + cust["customerID"].tolist()
    start = st.selectbox("Start from", ids)
    if start == "Typical customer":
        base = {c: cust[c].mode()[0] for c in core.CAT_COLS}
        base.update({"tenure": int(cust["tenure"].median()), "MonthlyCharges": float(cust["MonthlyCharges"].median()),
                     "TotalCharges": float(cust["TotalCharges"].median()), "SeniorCitizen": 0})
    else:
        base = cust.loc[cust["customerID"] == start, core.FEATURES].iloc[0].to_dict()

    with st.form("customer_form"):
        k = start  # widgets reset when the starting customer changes
        a, b, c = st.columns(3)
        vals = {}
        vals["tenure"] = a.slider("Tenure (months)", 0, 72, int(base["tenure"]), key=f"tenure_{k}")
        vals["MonthlyCharges"] = b.number_input("Monthly charges", 0.0, 200.0, float(base["MonthlyCharges"]), 0.5, key=f"mc_{k}")
        vals["TotalCharges"] = c.number_input("Total charges", 0.0, 10000.0, float(base["TotalCharges"]), 10.0, key=f"tc_{k}")
        vals["SeniorCitizen"] = 1 if a.selectbox("Senior citizen", ["No", "Yes"], index=int(base["SeniorCitizen"]), key=f"sc_{k}") == "Yes" else 0
        options = {col: sorted(cust[col].unique()) for col in core.CAT_COLS}
        cols = st.columns(3)
        for i, col in enumerate(core.CAT_COLS):
            opts = options[col]
            vals[col] = cols[i % 3].selectbox(col, opts, index=opts.index(base[col]), key=f"{col}_{k}")
        submitted = st.form_submit_button("Score this customer")

    row = pd.DataFrame([vals])[core.FEATURES]
    res = core.score_customer(art, row)
    st.write("")
    left, right = st.columns([1, 1.25])
    with left:
        tag = "tag-high" if res["flagged"] else "tag-low"
        status = "Flag for retention" if res["flagged"] else "Not flagged"
        st.markdown(f'<div class="kpi"><div class="l">Predicted churn risk</div><div class="v">{res["prob"] * 100:.0f}%</div>'
                    f'<div style="margin-top:8px;"><span class="tag {tag}">{status}</span></div></div>', unsafe_allow_html=True)
        st.pyplot(charts.risk_bar(res["prob"], thr))
        st.markdown(f"**Segment:** {res['segment']}")
        if res["flagged"]:
            st.markdown("**Suggested offers**")
            for o in res["offers"]:
                st.markdown(f"- {o}")
        else:
            st.markdown("Below the retention threshold: loyalty rewards, no discount needed.")
        drivers = core.risk_drivers(pd.Series(vals))
        st.markdown("**Main risk drivers present:** " + (", ".join(drivers) if drivers else "none of the main drivers"))
    with right:
        contrib = core.explain_customer(art, row)
        st.pyplot(charts.contribution_chart(contrib))
        st.caption("Each bar compares this customer with typical customers: the predicted risk minus the average risk when that one "
                   "feature is replaced by typical values. Orange pushes the risk up, navy pushes it down. "
                   "This explains the model, it does not prove cause.")


def page_why():
    header("Why customers leave", "What drives churn in the model",
           "The doctor's requirement: apply Explainable AI (XAI) to the chosen model. Two methods were used on the final Gradient Boosting model, so the result does not depend on a single technique.")
    t1, t2, t3 = st.tabs(["Most influential features", "Direction of the effect", "Evidence in the data"])
    with t1:
        a, b = st.columns(2)
        a.pyplot(charts.importance_bars(core.PERMUTATION_TOP, "Permutation importance (drop in PR-AUC)", "{:.3f}"))
        b.pyplot(charts.importance_bars(core.SHAP_TOP, "SHAP importance (mean absolute value)", "{:.2f}", color=charts.NAVY))
        st.markdown("Both methods agree on the top three: **Contract, tenure and InternetService** (rank correlation 0.79). "
                    "PaymentMethod and PaperlessBilling rank higher in SHAP than in permutation importance, probably because they share "
                    "information with other columns.")
        st.caption("Values are the results of the project notebook run on the test set.")
    with t2:
        st.table(pd.DataFrame({
            "Feature value": ["Month-to-month contract", "Fiber optic internet", "Electronic check", "Paperless billing", "High monthly charges",
                              "Long tenure", "Two-year contract", "No internet service", "TechSupport / OnlineSecurity"],
            "Effect on churn risk": ["Increases", "Increases", "Increases", "Increases", "Increases",
                                     "Decreases", "Decreases", "Decreases", "Decrease"],
        }))
        st.markdown("The model agrees with the exploratory analysis in **14 of 21** compared columns. None of the 7 disagreements "
                    "is among the 15 most influential columns.")
    with t3:
        a, b = st.columns(2)
        a.pyplot(charts.churn_bars(core.churn_by(cust[cust["InternetService"] == "Fiber optic"], "OnlineSecurity"),
                                   "Fiber customers: churn by OnlineSecurity", art["overall_churn"] * 100))
        b.pyplot(charts.churn_bars(core.churn_by(cust[cust["InternetService"] == "Fiber optic"], "TechSupport"),
                                   "Fiber customers: churn by TechSupport", art["overall_churn"] * 100))
        st.markdown("Within the same plan, customers with security or tech support churn about half as often. "
                    "Price alone is not the reason: inside each plan, churners pay the same or less than those who stay.")
    st.warning("XAI explains what the model relies on. It does not prove that a feature causes churn: that needs a pilot with a control group.")


def page_segments():
    header("Segments and offers", "Three customer groups, three actions",
           "K-Means grouped the customers by profile without seeing who churned. Churn was used afterwards only to describe the groups.")
    cols = st.columns(3)
    for col, (seg, row) in zip(cols, profile.iterrows()):
        bullets = (f"<li>{row['customers']:,.0f} customers ({row['share'] * 100:.0f}%)</li>"
                   f"<li>Churn {row['churn_rate'] * 100:.1f}% (lift {row['lift']:.2f})</li>"
                   f"<li>Tenure {row['avg_tenure']:.0f} months, monthly charge {row['avg_monthly']:.0f}</li>"
                   f"<li>{row['pct_month_to_month'] * 100:.0f}% month-to-month, {row['pct_fiber'] * 100:.0f}% Fiber, "
                   f"{row['pct_no_internet'] * 100:.0f}% no internet</li>")
        offers = "".join(f"<li>{o}</li>" for o in row["offers"])
        col.markdown(f'<div class="card"><h4>{row["name"]}</h4><ul style="padding-left:1.1rem;margin:6px 0;">{bullets}</ul>'
                     f'<div style="font-weight:700;color:{ORANGE};margin-top:8px;">Offer</div><ul style="padding-left:1.1rem;margin:4px 0;">{offers}</ul></div>',
                     unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    a.pyplot(charts.share_vs_churners(profile))
    top = profile.iloc[0]
    b.markdown(f"<div class='kpi' style='margin-top:30px;'><div class='v'>{top['lost_customers'] / profile['lost_customers'].sum() * 100:.0f}%</div>"
               f"<div class='l'>of all churners are in the group <b>{top['name']}</b>, which is only {top['share'] * 100:.0f}% of customers</div></div>",
               unsafe_allow_html=True)
    st.caption("Silhouette score of K-Means with k = 3 is about 0.20: the segments are a practical grouping, not sharply separated clusters. "
               "Offers come from rules tied to the EDA and XAI drivers; only groups with above-average churn get a retention offer. "
               "Their real effect has to be measured with a pilot.")


def page_model():
    header("Model and method", "How the model was built and how good it is")
    m = art["metrics"]
    c = st.columns(4)
    kpi(c[0], f"{m['recall'] * 100:.0f}%", "of churners caught (recall)")
    kpi(c[1], f"{m['precision'] * 100:.0f}%", "of flagged customers really churn (precision)")
    kpi(c[2], f"{m['pr_auc']:.3f}", f"PR-AUC (baseline {m['baseline_pr_auc']:.3f})")
    kpi(c[3], f"{m['roc_auc']:.3f}", "ROC-AUC (baseline 0.500)")
    st.write("")
    a, b = st.columns([1, 1.4])
    a.pyplot(charts.confusion_fig(m["tn"], m["fp"], m["fn"], m["tp"]))
    b.markdown(
        f"""
**Test set: {art['n_test']:,} customers, evaluated once.** At the threshold {thr:.3f} the model catches {m['tp']} of {m['tp'] + m['fn']} churners
and sends {m['fp']} unnecessary offers to loyal customers. Accuracy is {m['accuracy'] * 100:.1f}%, but a model that always says "stay" already
gets {m['baseline_accuracy'] * 100:.1f}% while catching nobody, so recall and PR-AUC are the metrics that matter.

**Pipeline:** 19 features, scaled and one-hot encoded into 26 inputs, inside one scikit-learn Pipeline so nothing leaks from validation
folds. 80/20 stratified split (random state 42).

**Model choice:** four models were compared with 5-fold cross-validation and tuned with GridSearchCV. After tuning they were statistically tied:
"""
    )
    st.table(pd.DataFrame({
        "Model": ["Gradient Boosting (chosen)", "Random Forest", "Logistic Regression", "KNN"],
        "CV PR-AUC": [0.667, 0.665, 0.660, 0.633],
        "Std": [0.021, 0.019, 0.020, 0.019],
    }))
    st.markdown("Gradient Boosting was selected by a rule fixed in advance (highest cross-validated PR-AUC). "
                "The threshold was chosen from out-of-fold predictions on the training set only.")
    st.markdown("**Limits:** fictional IBM sample data; XAI explains the model but does not prove cause; the threshold assumes a missed churner "
                "costs more than an unnecessary offer; lost revenue is not profit.")
    st.markdown("**Project team:** Rokaia Rezk, Rofaida Essam, Yomna Waleed, Roqia El-Gendy, Ganna Essa.")


{"Overview": page_overview, "Priority customers": page_priority, "Score a customer": page_score,
 "Why customers leave": page_why, "Segments and offers": page_segments, "Model and method": page_model}[page]()
