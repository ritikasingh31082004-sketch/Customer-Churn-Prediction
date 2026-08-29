"""
eda.py
------
Exploratory Data Analysis for the customer churn dataset.
Saves key charts to reports/figures/ for use in the README / report.

Run from project root:
    python notebooks/eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("data/customer_churn.csv")
FIG_DIR = "reports/figures"

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nChurn distribution:\n", df["churn"].value_counts())

# 1. Churn distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="churn", palette="Set2")
plt.title("Churn Distribution")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_distribution.png", dpi=120)
plt.close()

# 2. Churn rate by contract type
plt.figure(figsize=(6, 4))
rate = df.groupby("contract")["churn"].apply(lambda s: (s == "Yes").mean()).sort_values()
sns.barplot(x=rate.index, y=rate.values, palette="Set2")
plt.ylabel("Churn rate")
plt.title("Churn Rate by Contract Type")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_by_contract.png", dpi=120)
plt.close()

# 3. Tenure distribution by churn
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure_months", hue="churn", bins=30, kde=True, element="step")
plt.title("Tenure Distribution by Churn")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/tenure_by_churn.png", dpi=120)
plt.close()

# 4. Monthly charges by churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="churn", y="monthly_charges", palette="Set2")
plt.title("Monthly Charges by Churn")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/charges_by_churn.png", dpi=120)
plt.close()

# 5. Correlation heatmap (numeric features)
plt.figure(figsize=(6, 5))
numeric_df = df.copy()
numeric_df["churn_flag"] = (numeric_df["churn"] == "Yes").astype(int)
corr = numeric_df[["tenure_months", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen", "churn_flag"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=120)
plt.close()

print(f"\nSaved 5 figures to {FIG_DIR}/")
