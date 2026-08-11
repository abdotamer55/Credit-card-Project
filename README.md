# 💳 Credit Card Customer Segmentation

> **Unsupervised Machine Learning Project for Customer Behavior Analysis and Credit Card Segmentation**

This project analyzes credit card customer behavior and uses **K-Means clustering** to divide customers into meaningful behavioral segments.

The workflow covers the complete machine-learning analysis pipeline:

**Data Loading → Data Quality Analysis → Preprocessing → Outlier Treatment → EDA → Feature Engineering → Standardization → PCA → K-Means Clustering → Model Evaluation → Customer Profiling → Business Recommendations**

---

## 📌 Project Overview

Credit card customer datasets contain many behavioral variables such as:

- Account balance
- Purchase amount
- Purchase frequency
- Cash advance usage
- Credit limit
- Payment behavior
- Installment purchases
- Full-payment ratio
- Customer tenure

Analyzing all customers as one group can hide important behavioral differences.

The goal of this project is to identify **customer segments with similar financial and purchasing behavior** without using predefined target labels.

### 🎯 Main Objectives

1. Understand the structure and quality of the customer dataset.
2. Detect and handle missing values.
3. Detect and reduce the effect of extreme values.
4. Explore customer behavior through visual analysis.
5. Create meaningful behavioral features.
6. Standardize the feature space.
7. Reduce dimensionality using PCA.
8. Determine a suitable number of customer clusters.
9. Build a K-Means clustering model.
10. Evaluate clustering quality using multiple metrics.
11. Profile each customer segment.
12. Translate the clustering results into actionable business insights.

---

## 📊 Dataset

The project uses the **CC GENERAL** credit card customer dataset.

### Dataset Size

- **8,950 customers**
- **18 original columns**
- Customer identifier: `CUST_ID`
- 17 behavioral/numerical attributes

### Main Features

| Feature | Description |
|---|---|
| `CUST_ID` | Unique customer identifier |
| `BALANCE` | Customer account balance |
| `BALANCE_FREQUENCY` | Frequency of balance updates |
| `PURCHASES` | Total purchase amount |
| `ONEOFF_PURCHASES` | One-time purchase amount |
| `INSTALLMENTS_PURCHASES` | Installment purchase amount |
| `CASH_ADVANCE` | Cash advance amount |
| `PURCHASES_FREQUENCY` | Purchase frequency |
| `ONEOFF_PURCHASES_FREQUENCY` | Frequency of one-time purchases |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frequency of installment purchases |
| `CASH_ADVANCE_FREQUENCY` | Cash advance frequency |
| `CASH_ADVANCE_TRX` | Number of cash advance transactions |
| `PURCHASES_TRX` | Number of purchase transactions |
| `CREDIT_LIMIT` | Customer credit limit |
| `PAYMENTS` | Payment amount |
| `MINIMUM_PAYMENTS` | Minimum payment amount |
| `PRC_FULL_PAYMENT` | Percentage of full payments |
| `TENURE` | Customer relationship duration |

---

# 🔎 Project Workflow

## 1. Data Loading & Initial Exploration

The notebook starts by loading the `CC GENERAL.csv` dataset using Pandas.

Initial analysis includes:

- Dataset shape
- Column names
- Data types
- Statistical summary
- Missing-value analysis
- General data-quality inspection

The dataset contains **8,950 customer records and 18 columns**.

---

## 2. Data Preprocessing

### Missing Values

Two columns contain missing values:

- `MINIMUM_PAYMENTS` → **313 missing values**
- `CREDIT_LIMIT` → **1 missing value**

Missing values are handled using **median imputation**.

This approach keeps the dataset size unchanged while reducing the influence that extreme values can have on the replacement value.

### Customer ID

`CUST_ID` is removed from the modeling features because it is an identifier rather than a behavioral variable.

The original customer IDs are preserved separately so that clustering results can later be connected back to individual customers.

### Outlier Detection

The project uses the **Interquartile Range (IQR)** method:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Outliers are detected for every numerical feature.

Instead of deleting these observations, the project applies **winsorization/clipping**, limiting extreme observations to the calculated IQR boundaries.

This preserves the customer records while reducing the impact of extreme values on distance-based clustering.

---

# 📈 3. Exploratory Data Analysis

The EDA stage investigates customer behavior before clustering.

The notebook generates visualizations for:

- Feature distributions
- Correlation structure
- Purchase behavior
- Cash advance behavior
- Full-payment behavior
- Customer tenure
- Feature relationships

### Generated Visualizations

| Figure | Visualization |
|---|---|
| 01 | Feature distribution histograms |
| 02 | Correlation heatmap |
| 03 | Purchase behavior boxplots |
| 04 | Cash advance boxplots |
| 05 | Full payment distribution |
| 06 | Tenure distribution |
| 07 | Scatter plot matrix |

These visualizations help understand the underlying structure of customer behavior and identify relationships between variables.

---

# 🧩 4. Feature Engineering

Six behavioral features are engineered to capture relationships that are not directly represented by the original variables.

### 1. Purchase Ratio

```text
PURCHASE_RATIO =
PURCHASES / (PURCHASES + CASH_ADVANCE + 1)
```

Measures the relative importance of purchases compared with cash advances.

### 2. Credit Usage

```text
CREDIT_USAGE =
BALANCE / (CREDIT_LIMIT + 1)
```

Represents the customer's balance relative to their available credit.

### 3. Payment-to-Balance Ratio

```text
PAYMENT_TO_BALANCE =
PAYMENTS / (BALANCE + 1)
```

Provides an indication of payment behavior relative to outstanding balance.

### 4. Average Purchase per Transaction

```text
AVG_PURCHASE_PER_TRX =
PURCHASES / (PURCHASES_TRX + 1)
```

Estimates the average value of a purchase transaction.

### 5. Average Cash Advance per Transaction

```text
AVG_CASH_PER_TRX =
CASH_ADVANCE / (CASH_ADVANCE_TRX + 1)
```

Estimates the average cash advance transaction size.

### 6. Installment Tendency

```text
INSTALLMENT_TENDENCY =
INSTALLMENTS_PURCHASES / (PURCHASES + 1)
```

Captures the tendency toward installment-based purchases.

After feature engineering, the dataset contains **23 modeling features**.

---

# ⚖️ 5. Feature Standardization

Because clustering algorithms rely on distances, features with large numerical scales can dominate the results.

The project therefore applies:

```python
StandardScaler()
```

The resulting feature matrix has:

```text
8,950 customers × 23 standardized features
```

Standardization puts the features on a comparable scale before dimensionality reduction and clustering.

---

# 🧠 6. PCA Dimensionality Reduction

Principal Component Analysis (**PCA**) is used to reduce dimensionality while preserving most of the information contained in the standardized feature space.

The project retains **95% of the variance**.

### PCA Result

- Original features: **23**
- Components required for 95% variance: **13**

For visualization, a separate 2-component PCA representation is created.

The first two principal components explain approximately:

**49.6% of the total variance**

PCA is mainly used here to simplify the feature space and make clustering more manageable and interpretable.

---

# 🤖 7. K-Means Clustering

The main clustering algorithm is **K-Means**.

The notebook evaluates values of `K` from **2 to 10**.

Three clustering diagnostics are calculated:

- **Inertia**
- **Silhouette Score**
- **Calinski-Harabasz Index**

### K Evaluation

| K | Inertia | Silhouette | Calinski-Harabasz |
|---:|---:|---:|---:|
| 2 | 144866.44 | 0.2455 | 2689.42 |
| 3 | 121344.54 | 0.2401 | 2472.35 |
| 4 | 111768.47 | 0.2052 | 2044.74 |
| 5 | 103221.69 | 0.1905 | 1845.52 |
| 6 | 97267.56 | 0.1972 | 1676.11 |
| 7 | 91498.90 | 0.1859 | 1578.63 |
| 8 | 85513.21 | 0.1946 | 1537.08 |
| 9 | 81369.09 | 0.1978 | 1470.20 |
| 10 | 76100.26 | 0.1881 | 1465.95 |

The project selects:

```text
K = 4
```

based on the notebook's chosen elbow/cluster-interpretation approach.

---

# 📏 8. Model Evaluation

The final K-Means model is evaluated using three unsupervised clustering metrics.

### Final K-Means Metrics

| Metric | Score | Interpretation |
|---|---:|---|
| Silhouette Score | **0.2052** | Measures cluster separation and cohesion |
| Davies-Bouldin Index | **1.6158** | Lower values indicate better separation |
| Calinski-Harabasz Index | **2044.74** | Higher values generally indicate better-defined clusters |

The evaluation provides multiple perspectives instead of relying on a single clustering metric.

> **Important:** The Silhouette Score of 0.2052 indicates that the customer groups are not extremely well separated. Therefore, the segments should be interpreted as behavioral patterns rather than perfectly distinct customer populations.

---

# 👥 9. Customer Segmentation Results

The final model produces **4 customer segments**.

## Cluster 0 — Low-activity Customers

**2,215 customers — 24.7%**

Key characteristics:

- Average balance: **1,256.72**
- Average purchases: **77.68**
- Average cash advance: **1,029.67**
- Average credit limit: **2,755.09**
- Average payments: **942.12**
- Full-payment ratio: **0.03**

### Behavioral interpretation

This group has very low purchase activity but relatively noticeable cash-advance usage.

### Potential business direction

- Re-engagement campaigns
- Purchase incentives
- Targeted offers to convert customers from low purchase activity to regular card usage
- Responsible-credit communication around cash advances

---

## Cluster 1 — Frequent-purchase Customers

**1,810 customers — 20.2%**

Key characteristics:

- Average balance: **1,657.72**
- Average purchases: **2,162.92**
- Average cash advance: **280.79**
- Average credit limit: **6,432.59**
- Average payments: **2,420.72**
- Full-payment ratio: **0.13**

### Behavioral interpretation

This is the strongest purchase-oriented segment.

Customers show:

- High purchase volume
- High purchase frequency
- Strong installment activity
- Relatively low cash-advance usage

### Potential business direction

- Premium rewards
- Cashback campaigns
- Merchant partnerships
- Personalized purchase offers
- Loyalty programs

---

## Cluster 2 — Cash-advance Customers

**3,597 customers — 40.2%**

Key characteristics:

- Average balance: **468.44**
- Average purchases: **507.24**
- Average cash advance: **31.88**
- Average credit limit: **3,392.08**
- Average payments: **710.77**
- Full-payment ratio: **0.12**

### Behavioral interpretation

This is the **largest segment** in the final clustering solution.

The segment has relatively low balances and moderate purchase behavior, with very low average cash-advance usage according to the cluster averages produced by the notebook.

### Potential business direction

- Encourage broader card usage
- Cross-sell relevant products
- Increase transaction frequency through rewards
- Promote installment and merchant offers

---

## Cluster 3 — Steady-spending Customers

**1,328 customers — 14.8%**

Key characteristics:

- Average balance: **3,746.97**
- Average purchases: **516.56**
- Average cash advance: **2,345.61**
- Average credit limit: **7,232.27**
- Average payments: **2,282.73**
- Full-payment ratio: **0.03**

### Behavioral interpretation

This segment combines:

- High balances
- High credit limits
- High cash-advance activity
- Moderate purchase activity
- Low full-payment ratio

### Potential business direction

- Risk-aware customer management
- Payment behavior monitoring
- Personalized repayment incentives
- Careful credit-limit management
- Offers designed around responsible utilization

---

# 📊 Segment Distribution

| Cluster | Segment | Customers | Share |
|---:|---|---:|---:|
| 0 | Low-activity customers | 2,215 | 24.7% |
| 1 | Frequent-purchase customers | 1,810 | 20.2% |
| 2 | Cash-advance customers | 3,597 | 40.2% |
| 3 | Steady-spending customers | 1,328 | 14.8% |
| **Total** | | **8,950** | **100%** |

The largest segment is **Cluster 2**, representing **40.2%** of the customer base.

---

# 📊 10. Cluster Visualization

The notebook generates additional visual analysis for the final customer segments:

| Figure | Visualization |
|---|---|
| 10 | K-Means clustering visualization |
| 15 | Model evaluation |
| 16 | Cluster distribution + radar chart |
| 17 | Cluster feature comparison |
| 18 | Cluster feature heatmap |
| 19 | Tenure distribution by cluster |

These visualizations make the final customer segments easier to compare and interpret.

---

# 💾 11. Output Files

The project exports two main result datasets.

### `clustering_result.csv`

Contains:

- `CUST_ID`
- `Cluster`
- `Cluster_Name`

This file is useful when the goal is to obtain a simple customer-to-segment mapping.

### `data_with_clusters.csv`

Contains the processed customer dataset with the final cluster assignment.

This file is useful for:

- Further analysis
- Dashboard development
- Customer profiling
- Business intelligence
- Segment-level reporting

---

# 🗂️ Suggested Project Structure

```text
Credit-card-Project/
│
├── data/
│   └── CC GENERAL.csv
│
├── model/
│   ├── clustering_result.csv
│   └── data_with_clusters.csv
│
├── Visulization/
│   ├── 01_feature_distribution_histograms.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_purchase_behavior_boxplots.png
│   ├── 04_cash_advance_boxplots.png
│   ├── 05_full_payment_distribution.png
│   ├── 06_tenure_distribution.png
│   ├── 07_scatter_plot_matrix.png
│   ├── 08_PCAanalysis.png
│   ├── 09_K_value_evaluation.png
│   ├── 10_*.png
│   ├── 15_model_evaluation.png
│   ├── 16_cluster_distribution_and_radar.png
│   ├── 17_*.png
│   ├── 18_*.png
│   └── 19_*.png
│
├── credit_card.ipynb
└── README.md
```

> The exact visualization filenames for figures 10, 11–14, and 17–19 depend on the notebook's generated outputs.

---

# 🛠️ Technologies & Libraries

The project is implemented in Python using:

- **Python**
- **NumPy** — numerical operations
- **Pandas** — data manipulation and analysis
- **Matplotlib** — visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing, PCA, clustering, and evaluation
- **SciPy** — hierarchical-analysis utilities and dendrogram support

### Main Machine Learning Components

```python
StandardScaler
PCA
KMeans
silhouette_score
davies_bouldin_score
calinski_harabasz_score
```

---

# ▶️ How to Run

## 1. Clone the project

```bash
git clone <your-repository-url>
cd Credit-card-Project
```

## 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

## 3. Place the dataset

Make sure the dataset is available at:

```text
data/CC GENERAL.csv
```

## 4. Open the notebook

```bash
jupyter notebook credit_card.ipynb
```

or:

```bash
jupyter lab
```

## 5. Run all cells

Run the notebook from top to bottom.

The workflow will:

1. Load the data
2. Inspect the dataset
3. Clean missing values
4. Handle outliers
5. Perform EDA
6. Engineer features
7. Standardize the data
8. Apply PCA
9. Evaluate K values
10. Train K-Means
11. Evaluate the clustering
12. Analyze customer profiles
13. Export clustering results
14. Generate business recommendations

---

# 💼 Business Value

This project demonstrates how unsupervised machine learning can transform raw credit-card transaction data into **actionable customer segments**.

Instead of treating every customer identically, a financial institution can use these segments to support:

### 🎯 Customer Marketing

Different customers can receive different campaigns based on their spending behavior.

### 💳 Personalized Offers

Frequent purchasers can receive rewards, cashback, and merchant-specific offers.

### 📈 Customer Engagement

Low-activity customers can be targeted with campaigns designed to increase card usage.

### ⚠️ Risk-Aware Management

Customers with high balances and high cash-advance activity can receive more targeted payment and credit-management strategies.

### 🤝 Customer Relationship Management

Customer segments can be incorporated into CRM systems and dashboards to support more personalized decision-making.

---

# 🧠 Key Insights

The analysis reveals four broad behavioral patterns:

```text
                    CUSTOMER BASE
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    Low Activity    Frequent Buy    High Balance
          │              │              │
     Cluster 0       Cluster 1       Cluster 3
          │
          └──────────────┐
                         │
                   Cluster 2
                 Largest Segment
                    40.2%
```

### Most important findings

- The customer base contains **8,950 customers**.
- Six engineered behavioral features are added to the original variables.
- The final modeling dataset contains **23 features**.
- PCA requires **13 components** to retain 95% variance.
- The first two PCA components explain approximately **49.6%** of the variance.
- The final K-Means solution uses **4 clusters**.
- Cluster 2 is the largest segment at **40.2%**.
- Cluster 1 is strongly purchase-oriented.
- Cluster 3 combines high balances, high credit limits, and high cash-advance activity.
- Cluster 0 shows very low purchase activity.
- The final clustering has a **Silhouette Score of 0.2052**, so the segments should be treated as useful behavioral groupings rather than perfectly separated populations.

---

# ⚠️ Limitations

This project is an exploratory customer-segmentation analysis and has several limitations.

### 1. Unsupervised Learning

There is no predefined target variable, so the clusters represent discovered patterns rather than objectively labeled customer types.

### 2. Cluster Quality

The final Silhouette Score is relatively modest (**0.2052**), indicating overlap between some customer groups.

### 3. Fixed K Selection

The notebook selects `K=4` based on the chosen elbow/interpretation approach, even though some evaluation metrics show stronger values for smaller K values.

### 4. Behavioral Data Only

The segmentation is based on the available credit-card behavioral variables. It does not include demographic, geographic, income, or external financial information.

### 5. Business Recommendations

The recommendations are analytical interpretations of the cluster profiles. They should be validated with domain experts and real business KPIs before being used in production.

---

# 🚀 Possible Future Improvements

The project can be extended in several directions:

- Compare K-Means with hierarchical clustering.
- Experiment with DBSCAN.
- Optimize cluster selection using multiple criteria.
- Compare PCA-based clustering with clustering on the full standardized feature space.
- Perform deeper statistical profiling of each segment.
- Build a customer-segmentation dashboard using Power BI or Streamlit.
- Add demographic and transactional features when available.
- Create automated segment-based marketing rules.
- Track cluster stability across different samples and time periods.
- Test alternative scaling and transformation strategies.
- Build a production pipeline that assigns new customers to existing segments.

---

# 📚 Project Skills Demonstrated

This project demonstrates practical experience in:

**Data Analysis**
- Data inspection
- Missing-value analysis
- Descriptive statistics
- Correlation analysis

**Data Preprocessing**
- Median imputation
- IQR outlier detection
- Winsorization
- Feature standardization

**Feature Engineering**
- Ratio features
- Behavioral metrics
- Transaction-level derived indicators

**Machine Learning**
- Unsupervised learning
- K-Means clustering
- PCA
- Cluster evaluation

**Data Visualization**
- Histograms
- Boxplots
- Heatmaps
- Scatter plots
- Elbow curves
- Radar charts
- Cluster comparison charts

**Business Analytics**
- Customer profiling
- Customer segmentation
- Segment interpretation
- Business recommendations

---

# 👨‍💻 Author

**Abdelrahman Tamer**

Data Analysis & AI Enthusiast

---

## ⭐ Project Summary

> **This project applies an end-to-end unsupervised machine learning workflow to segment 8,950 credit card customers into four behavioral groups using data preprocessing, feature engineering, PCA, and K-Means clustering. The final analysis converts customer transaction patterns into interpretable segments and business-oriented recommendations.**
