from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "CC GENERAL.csv"

CLUSTER_NAMES = {
    0: "Low-activity customers",
    1: "Frequent-purchase customers",
    2: "Cash-advance customers",
    3: "Steady-spending customers",
}

KEY_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT",
]


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Put 'CC GENERAL.csv' inside the project's data folder."
        )
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_segmentation_model():
    """Rebuild the same preprocessing + PCA + K-Means workflow used in the notebook."""
    df = load_data().copy()

    customer_ids = df["CUST_ID"].copy()

    # Same missing-value treatment as the notebook.
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
        df["MINIMUM_PAYMENTS"].median()
    )
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(
        df["CREDIT_LIMIT"].median()
    )

    df = df.drop(columns=["CUST_ID"])

    # Save training clipping bounds so new customers use the same preprocessing.
    clip_bounds = {}
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clip_bounds[col] = (lower, upper)
        df[col] = df[col].clip(lower, upper)

    # Same six engineered features as the notebook.
    df["PURCHASE_RATIO"] = df["PURCHASES"] / (
        df["PURCHASES"] + df["CASH_ADVANCE"] + 1
    )
    df["CREDIT_USAGE"] = df["BALANCE"] / (df["CREDIT_LIMIT"] + 1)
    df["PAYMENT_TO_BALANCE"] = df["PAYMENTS"] / (df["BALANCE"] + 1)
    df["AVG_PURCHASE_PER_TRX"] = df["PURCHASES"] / (
        df["PURCHASES_TRX"] + 1
    )
    df["AVG_CASH_PER_TRX"] = df["CASH_ADVANCE"] / (
        df["CASH_ADVANCE_TRX"] + 1
    )
    df["INSTALLMENT_TENDENCY"] = df["INSTALLMENTS_PURCHASES"] / (
        df["PURCHASES"] + 1
    )

    feature_columns = df.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    labeled = load_data().copy()
    labeled["Cluster"] = labels
    labeled["Cluster_Name"] = labeled["Cluster"].map(CLUSTER_NAMES)

    return {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "clip_bounds": clip_bounds,
        "feature_columns": feature_columns,
        "labels": labels,
        "labeled": labeled,
    }


def prepare_new_customer(values: dict, model_bundle: dict):
    """Apply the notebook's feature engineering to one new customer."""
    row = pd.DataFrame([values])

    # Match training preprocessing.
    for col, (lower, upper) in model_bundle["clip_bounds"].items():
        if col in row.columns:
            row[col] = row[col].clip(lower, upper)

    row["PURCHASE_RATIO"] = row["PURCHASES"] / (
        row["PURCHASES"] + row["CASH_ADVANCE"] + 1
    )
    row["CREDIT_USAGE"] = row["BALANCE"] / (row["CREDIT_LIMIT"] + 1)
    row["PAYMENT_TO_BALANCE"] = row["PAYMENTS"] / (row["BALANCE"] + 1)
    row["AVG_PURCHASE_PER_TRX"] = row["PURCHASES"] / (
        row["PURCHASES_TRX"] + 1
    )
    row["AVG_CASH_PER_TRX"] = row["CASH_ADVANCE"] / (
        row["CASH_ADVANCE_TRX"] + 1
    )
    row["INSTALLMENT_TENDENCY"] = row["INSTALLMENTS_PURCHASES"] / (
        row["PURCHASES"] + 1
    )

    row = row[model_bundle["feature_columns"]]
    X_scaled = model_bundle["scaler"].transform(row)
    X_pca = model_bundle["pca"].transform(X_scaled)

    cluster = int(model_bundle["kmeans"].predict(X_pca)[0])
    return cluster, X_pca[0]