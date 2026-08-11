import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from assets.style import load_css

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common import load_data

load_css()

st.set_page_config(
    page_title="EDA | Credit Card Segmentation",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Exploratory Data Analysis")
st.caption("Interactive exploration of the raw credit-card customer dataset.")

try:
    df = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

# -----------------------------
# Overview
# -----------------------------
st.markdown("## 📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Original Features", df.shape[1])
c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

with st.expander("View raw data"):
    st.dataframe(df.head(100), use_container_width=True)

# -----------------------------
# Missing values
# -----------------------------
st.markdown("## 🧹 Data Quality")

missing = (
    df.isna()
    .sum()
    .rename("Missing Count")
    .to_frame()
)
missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
missing = missing[missing["Missing Count"] > 0].sort_values(
    "Missing Count", ascending=False
)

if missing.empty:
    st.success("No missing values found.")
else:
    st.dataframe(missing, use_container_width=True)

# -----------------------------
# Distribution
# -----------------------------
st.markdown("## 📈 Feature Distribution")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
selected = st.selectbox(
    "Select a numerical feature",
    numeric_cols,
    index=numeric_cols.index("PURCHASES") if "PURCHASES" in numeric_cols else 0,
)

fig, ax = plt.subplots(figsize=(10, 4.5))
sns.histplot(df[selected].dropna(), bins=50, kde=True, ax=ax)
ax.set_title(f"Distribution of {selected}")
ax.set_xlabel(selected)
ax.set_ylabel("Customers")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# -----------------------------
# Correlation
# -----------------------------
st.markdown("## 🔗 Correlation Analysis")

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    corr,
    cmap="RdBu_r",
    center=0,
    ax=ax,
    linewidths=0.3,
)
ax.set_title("Feature Correlation Heatmap")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# -----------------------------
# Purchase behavior
# -----------------------------
st.markdown("## 🛒 Purchase Behavior")

purchase_cols = [
    c for c in [
        "PURCHASES",
        "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES",
        "PURCHASES_FREQUENCY",
    ] if c in df.columns
]

choice = st.multiselect(
    "Select purchase features",
    purchase_cols,
    default=purchase_cols,
)

if choice:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df[choice], ax=ax)
    ax.set_title("Purchase Behavior")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# -----------------------------
# Cash advance
# -----------------------------
st.markdown("## 💵 Cash Advance Behavior")

cash_cols = [
    c for c in [
        "CASH_ADVANCE",
        "CASH_ADVANCE_FREQUENCY",
        "CASH_ADVANCE_TRX",
    ] if c in df.columns
]

cash_choice = st.selectbox("Cash advance feature", cash_cols)

fig, ax = plt.subplots(figsize=(10, 4.5))
sns.histplot(df[cash_choice].dropna(), bins=50, kde=True, ax=ax)
ax.set_title(f"{cash_choice} Distribution")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# -----------------------------
# Relationships
# -----------------------------
st.markdown("## 🔍 Customer Relationships")

x_col = st.selectbox(
    "X-axis",
    numeric_cols,
    index=numeric_cols.index("BALANCE") if "BALANCE" in numeric_cols else 0,
)
y_col = st.selectbox(
    "Y-axis",
    numeric_cols,
    index=numeric_cols.index("PURCHASES") if "PURCHASES" in numeric_cols else 0,
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df[x_col], df[y_col], alpha=0.25, s=10)
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title(f"{x_col} vs {y_col}")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# -----------------------------
# Tenure
# -----------------------------
st.markdown("## 📅 Customer Tenure")

if "TENURE" in df.columns:
    tenure_counts = df["TENURE"].value_counts().sort_index()
    st.bar_chart