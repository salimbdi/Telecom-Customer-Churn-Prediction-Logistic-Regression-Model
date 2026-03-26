"""
train.py — Trains the Churn Prediction model and saves it to disk.

This script replicates the preprocessing and training logic from the notebook.
Run: python train.py
Output: model.pkl and dv.pkl in the current directory.
"""

import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Multiple mirrors — tried in order until one succeeds
DATA_URLS = [
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/mattborghi/EngineML/master/1_Classification/Telecom%20Customer%20Churn%20Prediction/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

def load_data():
    if not os.path.exists(DATA_FILE):
        print("→ Dataset not found locally. Trying to download...")
        downloaded = False
        for i, url in enumerate(DATA_URLS, 1):
            try:
                print(f"  Attempt {i}/{len(DATA_URLS)}: {url[:60]}...")
                urllib.request.urlretrieve(url, DATA_FILE)
                print(f"✓ Downloaded and saved as {DATA_FILE}")
                downloaded = True
                break
            except Exception as e:
                print(f"  ✗ Failed ({e})")

        if not downloaded:
            raise FileNotFoundError(
                "\n❌ All download mirrors failed.\n"
                "Please download the dataset manually from Kaggle:\n"
                "  https://www.kaggle.com/blastchar/telco-customer-churn\n"
                f"Then place '{DATA_FILE}' in this folder and re-run.\n"
            )
    else:
        print(f"✓ Found local dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
# 2. Preprocessing (same as notebook)
# ─────────────────────────────────────────────
# Categorical columns used for training
CATEGORICAL = [
    "gender", "seniorcitizen", "partner", "dependents",
    "phoneservice", "multiplelines", "internetservice",
    "onlinesecurity", "onlinebackup", "deviceprotection",
    "techsupport", "streamingtv", "streamingmovies",
    "contract", "paperlessbilling", "paymentmethod",
]

# Numerical columns
NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase all column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Lowercase all string values
    str_cols = df.dtypes[df.dtypes == "object"].index
    for col in str_cols:
        df[col] = df[col].str.lower().str.replace(" ", "_")

    # TotalCharges has some spaces — convert to numeric
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"] = df["totalcharges"].fillna(0)

    # Encode target
    df["churn"] = (df["churn"] == "yes").astype(int)

    return df


# ─────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────
def train():
    print("\n── Churn Prediction Model Training ──")

    df = load_data()
    df = preprocess(df)

    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    y_train = df_train["churn"].values
    y_test  = df_test["churn"].values

    # Feature dicts
    train_dicts = df_train[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    test_dicts  = df_test[CATEGORICAL + NUMERICAL].to_dict(orient="records")

    # Vectorize
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_test  = dv.transform(test_dicts)

    # Train Logistic Regression
    model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"\n✓ Model trained! AUC on test set: {auc:.4f}")

    # Save model + vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("dv.pkl", "wb") as f:
        pickle.dump(dv, f)

    print("✓ Saved: model.pkl")
    print("✓ Saved: dv.pkl")
    print("\n── Training complete ──\n")

    return model, dv


if __name__ == "__main__":
    train()
