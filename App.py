from pathlib import Path
import streamlit as st

from assets.style import load_css


# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the project's single source of styling.
# No page-level CSS is used here to avoid conflicts with Style.py.
load_css()


# =====================================
# Paths
# =====================================

BASE_DIR = Path(__file__).resolve().parent
BANNER_PATH = BASE_DIR / "Visulization" / "banner.jpg"
README_PATH = BASE_DIR / "README.md"


# =====================================
# Sidebar
# =====================================

with st.sidebar:
    st.markdown("## 💳 Credit Card ML")
    st.caption("Customer Segmentation Dashboard")

    st.divider()

    st.markdown("### 📊 Project Overview")
    st.write("• 8,950 Customers")
    st.write("• 23 Modeling Features")
    st.write("• 4 Customer Segments")
    st.write("• PCA — 95% Variance")

    st.divider()

    st.markdown("### 🤖 Machine Learning")
    st.write("• Feature Engineering")
    st.write("• StandardScaler")
    st.write("• PCA")
    st.write("• K-Means Clustering")

    st.divider()

    st.markdown("### 📈 Model Evaluation")
    st.write("Silhouette: **0.2052**")
    st.write("Davies-Bouldin: **1.6158**")
    st.write("Calinski-Harabasz: **2044.74**")

    st.divider()

    st.caption("Use the navigation menu above to explore EDA, Visualizations and Prediction.")


# =====================================
# Hero Section
# =====================================

if BANNER_PATH.exists():
    st.image(
        str(BANNER_PATH),
        use_container_width=True,
    )
else:
    st.info(
        "💡 Banner image not found. "
        "Add your banner at: images/banner.jpg"
    )

st.title("💳 Credit Card Customer Segmentation")

st.markdown(
    """
    An end-to-end **Unsupervised Machine Learning** project for analyzing
    credit card customer behavior and discovering meaningful customer
    segments using **K-Means Clustering**.
    """
)


# =====================================
# Project at a Glance
# =====================================

st.header("📊 Project at a Glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", "8,950")

with col2:
    st.metric("Modeling Features", "23")

with col3:
    st.metric("Final Clusters", "4")

with col4:
    st.metric("PCA Variance", "95%")


# =====================================
# Customer Segments
# =====================================

st.header("👥 Customer Segments")

seg1, seg2, seg3, seg4 = st.columns(4)

with seg1:
    st.subheader("Cluster 0")
    st.write("**Low-activity Customers**")
    st.metric("Customers", "2,215")
    st.caption("24.7% of customer base")

with seg2:
    st.subheader("Cluster 1")
    st.write("**Frequent-purchase Customers**")
    st.metric("Customers", "1,810")
    st.caption("20.2% of customer base")

with seg3:
    st.subheader("Cluster 2")
    st.write("**Cash-advance Customers**")
    st.metric("Customers", "3,597")
    st.caption("40.2% of customer base")

with seg4:
    st.subheader("Cluster 3")
    st.write("**Steady-spending Customers**")
    st.metric("Customers", "1,328")
    st.caption("14.8% of customer base")


# =====================================
# Model Performance
# =====================================

st.header("🤖 Clustering Model Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Silhouette Score",
        "0.2052",
        help="Measures cluster cohesion and separation.",
    )

with metric2:
    st.metric(
        "Davies-Bouldin Index",
        "1.6158",
        help="Lower values generally indicate better separation.",
    )

with metric3:
    st.metric(
        "Calinski-Harabasz Index",
        "2044.74",
        help="Higher values generally indicate better-defined clusters.",
    )


# =====================================
# Project Workflow
# =====================================

st.header("🔄 Project Workflow")

workflow = st.columns(7)

workflow_items = [
    ("01", "Data\nLoading"),
    ("02", "EDA"),
    ("03", "Preprocessing"),
    ("04", "Feature\nEngineering"),
    ("05", "PCA"),
    ("06", "K-Means"),
    ("07", "Segmentation"),
]

for column, (number, title) in zip(workflow, workflow_items):
    with column:
        st.subheader(number)
        st.caption(title)


# =====================================
# Quick Insights
# =====================================

st.header("🧠 Key Insights")

insight1, insight2 = st.columns(2)

with insight1:
    st.markdown("### 🛒 Largest Segment")
    st.write(
        "Cluster 2 is the largest customer group, containing "
        "**3,597 customers (40.2%)**."
    )

    st.markdown("### 💳 Purchase-oriented Segment")
    st.write(
        "Cluster 1 contains **1,810 customers** and represents the "
        "strongest purchase-oriented segment."
    )

with insight2:
    st.markdown("### 📉 Low-activity Segment")
    st.write(
        "Cluster 0 contains **2,215 customers** with relatively low "
        "purchase activity."
    )

    st.markdown("### 💰 High-balance Segment")
    st.write(
        "Cluster 3 contains **1,328 customers** characterized by "
        "higher balances and credit limits."
    )


# =====================================
# Navigation
# =====================================

st.header("🚀 Explore the Project")

nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.subheader("🔎 EDA")
    st.write(
        "Explore the raw dataset, missing values, distributions, "
        "correlations and customer behavior."
    )

with nav2:
    st.subheader("📊 Visualizations")
    st.write(
        "Explore the PCA, K-Means, cluster distribution and "
        "feature comparison visualizations."
    )

with nav3:
    st.subheader("🎯 Prediction")
    st.write(
        "Enter customer behavior and assign the customer to one "
        "of the four learned behavioral segments."
    )


# =====================================
# README Documentation
# =====================================

st.header("📚 Project Documentation")

if README_PATH.exists():
    readme_content = README_PATH.read_text(encoding="utf-8")

    # Avoid showing a duplicate banner if the README contains one.
    readme_content = readme_content.replace(
        '<p align="center"><img src="images/banner.jpg" alt="Insurance Charges Prediction Banner" width="100%"></p>',
        "",
    )
    readme_content = readme_content.replace(
        '<p align="center"> <img src="images/banner.jpg" alt="Insurance Charges Prediction Banner" width="100%"> </p>',
        "",
    )
    readme_content = readme_content.replace(
        '<p align="center"><img src="images/banner.jpg" alt="Credit Card Customer Segmentation Banner" width="100%"></p>',
        "",
    )
    readme_content = readme_content.replace(
        '<p align="center"> <img src="images/banner.jpg" alt="Credit Card Customer Segmentation Banner" width="100%"> </p>',
        "",
    )

    with st.expander("📖 Read Full Project README", expanded=False):
        st.markdown(readme_content)

else:
    st.warning(
        "README.md was not found. Put README.md in the same folder as App.py."
    )


# =====================================
# Footer
# =====================================

st.divider()

st.caption(
    "Credit Card Customer Segmentation • "
    "Unsupervised Machine Learning • "
    "Developed by Abdelrahman Tamer"
)