import sys
from pathlib import Path
import streamlit as st
from assets.style import load_css

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common import CLUSTER_NAMES, load_data, prepare_new_customer, train_segmentation_model

st.set_page_config(
    page_title="Customer Segmentation | Prediction",
    page_icon="🎯",
    layout="wide",
)

load_css()

st.title("🎯 Customer Segmentation Prediction")
st.caption(
    "Enter a customer's behavior and the trained K-Means model will assign "
    "the customer to one of the four discovered behavioral segments."
)

st.info(
    "This is an unsupervised-learning prediction page: the model predicts "
    "the customer's **cluster/segment**, not a credit approval or financial risk score."
)

try:
    bundle = train_segmentation_model()
except Exception as e:
    st.error(str(e))
    st.stop()

st.markdown("## 👤 Customer Profile")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        balance = st.number_input("Balance", min_value=0.0, value=1000.0, step=100.0)
        balance_frequency = st.number_input(
            "Balance Frequency", min_value=0.0, max_value=1.0, value=0.8, step=0.01
        )
        purchases = st.number_input("Purchases", min_value=0.0, value=1000.0, step=100.0)
        oneoff_purchases = st.number_input(
            "One-off Purchases", min_value=0.0, value=500.0, step=50.0
        )
        installments_purchases = st.number_input(
            "Installment Purchases", min_value=0.0, value=300.0, step=50.0
        )
        cash_advance = st.number_input(
            "Cash Advance", min_value=0.0, value=300.0, step=50.0
        )

    with col2:
        purchases_frequency = st.number_input(
            "Purchases Frequency", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
        oneoff_purchases_frequency = st.number_input(
            "One-off Purchase Frequency", min_value=0.0, max_value=1.0, value=0.2, step=0.01
        )
        purchases_installments_frequency = st.number_input(
            "Installment Purchase Frequency", min_value=0.0, max_value=1.0, value=0.3, step=0.01
        )
        cash_advance_frequency = st.number_input(
            "Cash Advance Frequency", min_value=0.0, max_value=1.5, value=0.1, step=0.01
        )
        cash_advance_trx = st.number_input(
            "Cash Advance Transactions", min_value=0.0, value=2.0, step=1.0
        )
        purchases_trx = st.number_input(
            "Purchase Transactions", min_value=0.0, value=15.0, step=1.0
        )

    with col3:
        credit_limit = st.number_input(
            "Credit Limit", min_value=50.0, value=4000.0, step=100.0
        )
        payments = st.number_input(
            "Payments", min_value=0.0, value=1500.0, step=100.0
        )
        minimum_payments = st.number_input(
            "Minimum Payments", min_value=0.0, value=500.0, step=50.0
        )
        prc_full_payment = st.number_input(
            "Full Payment Ratio", min_value=0.0, max_value=1.0, value=0.10, step=0.01
        )
        tenure = st.number_input(
            "Tenure (months)", min_value=1.0, max_value=24.0, value=12.0, step=1.0
        )

    submitted = st.form_submit_button(
        "🚀 Predict Customer Segment",
        use_container_width=True,
    )

if submitted:
    values = {
        "BALANCE": balance,
        "BALANCE_FREQUENCY": balance_frequency,
        "PURCHASES": purchases,
        "ONEOFF_PURCHASES": oneoff_purchases,
        "INSTALLMENTS_PURCHASES": installments_purchases,
        "CASH_ADVANCE": cash_advance,
        "PURCHASES_FREQUENCY": purchases_frequency,
        "ONEOFF_PURCHASES_FREQUENCY": oneoff_purchases_frequency,
        "PURCHASES_INSTALLMENTS_FREQUENCY": purchases_installments_frequency,
        "CASH_ADVANCE_FREQUENCY": cash_advance_frequency,
        "CASH_ADVANCE_TRX": cash_advance_trx,
        "PURCHASES_TRX": purchases_trx,
        "CREDIT_LIMIT": credit_limit,
        "PAYMENTS": payments,
        "MINIMUM_PAYMENTS": minimum_payments,
        "PRC_FULL_PAYMENT": prc_full_payment,
        "TENURE": tenure,
    }

    try:
        cluster, pca_point = prepare_new_customer(values, bundle)
        name = CLUSTER_NAMES[cluster]

        st.success(f"Customer assigned to **Cluster {cluster} — {name}**")

        result1, result2, result3 = st.columns(3)
        result1.metric("Predicted Cluster", cluster)
        result2.metric("Customer Segment", name)
        result3.metric(
            "PCA Position",
            f"PC1 {pca_point[0]:.2f} | PC2 {pca_point[1]:.2f}",
        )

        # Segment-level profile from the training data.
        segment_df = bundle["labeled"]
        segment = segment_df[segment_df["Cluster"] == cluster]

        st.markdown("## 📊 Segment Profile")

        profile_cols = st.columns(4)

        metrics = [
            ("Customers", len(segment)),
            ("Avg Purchases", segment["PURCHASES"].mean()),
            ("Avg Cash Advance", segment["CASH_ADVANCE"].mean()),
            ("Avg Credit Limit", segment["CREDIT_LIMIT"].mean()),
        ]

        for col, (label, value) in zip(profile_cols, metrics):
            with col:
                if label == "Customers":
                    st.metric(label, f"{int(value):,}")
                else:
                    st.metric(label, f"{value:,.2f}")

        st.markdown("### 🔎 Similar Segment Characteristics")
        st.dataframe(
            segment[
                [
                    "BALANCE",
                    "PURCHASES",
                    "CASH_ADVANCE",
                    "CREDIT_LIMIT",
                    "PAYMENTS",
                    "PURCHASES_FREQUENCY",
                    "PRC_FULL_PAYMENT",
                ]
            ].mean().to_frame("Segment Mean").round(2),
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------
# Model information
# -----------------------------
with st.expander("ℹ️ How the prediction works"):
    st.markdown(
        """
        1. The customer inputs are validated and preprocessed.
        2. Missing-value handling and IQR clipping follow the training workflow.
        3. Six behavioral features are engineered.
        4. `StandardScaler` transforms the features.
        5. PCA projects the customer into the trained 95%-variance space.
        6. K-Means assigns the customer to one of the four learned clusters.
        """
    )