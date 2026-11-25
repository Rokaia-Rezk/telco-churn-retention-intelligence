# Telco Churn Retention Dashboard

A Streamlit app that turns the Telco Churn project into a tool for a retention / customer-care manager:

| Page | What it gives the manager |
|---|---|
| Overview | Churn rate, revenue at risk, churn by tenure / contract / internet / payment |
| Priority customers | Active customers ranked by churn risk, with the risk drivers present and the suggested offer; CSV download |
| Score a customer | Risk for one customer, the reasons behind it and the offer |
| Why customers leave | XAI results (permutation importance + SHAP) and the evidence in the data |
| Segments and offers | The 3 K-Means customer groups, their churn and the offer for each |
| Model and method | Test results, confusion matrix, model choice, limits |

The model, the threshold (0.332) and the segments are rebuilt from `data/telco_churn.csv` on first load (about 10 seconds)
with exactly the same steps as the project notebook: same cleaning, 80/20 stratified split (random_state=42), same preprocessing
Pipeline, Gradient Boosting (200 trees, depth 2, learning rate 0.05), threshold from out-of-fold predictions, K-Means with k = 3.
The data is the public IBM Telco sample (a fictional company), so this is a demonstration of the method.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Create a new GitHub repository (for example `telco-churn-dashboard`) and upload everything in this folder
   (`app.py`, `core.py`, `charts.py`, `requirements.txt`, `.streamlit/config.toml`, `data/telco_churn.csv`, `README.md`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** -> **Deploy a public app from GitHub**.
4. Choose the repository, branch `main`, and main file path `app.py`.
5. Press **Deploy**. The first start installs the packages and trains the model (1-2 minutes). You get a public link to put on your CV / LinkedIn.

## Files

- `app.py` - the interface (pages, sidebar, styling)
- `core.py` - cleaning, model, threshold, segments, explanations (no Streamlit code)
- `charts.py` - charts in the project palette
- `.streamlit/config.toml` - theme (navy / orange / cream, same as the presentation)
- `data/telco_churn.csv` - the dataset
- `requirements.txt` - dependencies

## Notes

- Scores shown for customers are honest: out-of-fold predictions for training customers, held-out predictions for test customers.
- "What moves this customer's risk" compares a customer with 150 typical training customers, one feature at a time.
  It explains the model; it does not prove that a feature causes churn.
- Built by Rokaia Rezk - [LinkedIn](https://www.linkedin.com/in/rokaia-rezk-0761052bb/) - [GitHub](https://github.com/Rokaia-Rezk)
