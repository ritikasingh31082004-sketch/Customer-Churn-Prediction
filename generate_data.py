"""
generate_data.py
-----------------
Generates a realistic synthetic telecom customer-churn dataset.

The features are drawn from distributions and correlations that mimic
real churn drivers (long-known from telecom analytics: tenure, contract
type, monthly charges, support-call frequency, etc.), so the resulting
classification task has genuine, learnable signal rather than pure noise.

Run:
    python generate_data.py
Produces:
    customer_churn.csv  (5,000 rows)
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 5000

rng = np.random.default_rng(RANDOM_SEED)


def generate_dataset(n=N_SAMPLES):
    customer_id = [f"CUST-{10000 + i}" for i in range(n)]

    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])

    tenure_months = rng.gamma(shape=2.0, scale=15, size=n).clip(0, 72).astype(int)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20]
    )

    monthly_charges = np.where(
        internet_service == "Fiber optic",
        rng.normal(85, 15, n),
        np.where(internet_service == "DSL", rng.normal(55, 12, n), rng.normal(25, 8, n)),
    ).clip(18, 130)

    total_charges = (monthly_charges * tenure_months * rng.uniform(0.9, 1.05, n)).clip(0)

    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
    online_security = rng.choice(["Yes", "No"], size=n, p=[0.38, 0.62])
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])

    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n,
        p=[0.35, 0.2, 0.225, 0.225],
    )

    num_support_calls = rng.poisson(lam=1.5, size=n)
    num_support_calls = np.where(
        contract == "Month-to-month",
        num_support_calls + rng.poisson(lam=1.0, size=n),
        num_support_calls,
    )

    # --- Latent churn probability, built from realistic weighted drivers ---
    logit = (
        -1.6
        + 1.30 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        - 0.03 * tenure_months
        + 0.012 * (monthly_charges - 60)
        + 0.22 * num_support_calls
        - 0.55 * (tech_support == "Yes")
        - 0.35 * (online_security == "Yes")
        + 0.30 * (payment_method == "Electronic check")
        + 0.20 * (internet_service == "Fiber optic")
        + 0.15 * senior_citizen
        + rng.normal(0, 0.6, n)  # noise so the task isn't trivially separable
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "tenure_months": tenure_months,
            "contract": contract,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "num_support_calls": num_support_calls,
            "churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "customer_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["churn"].value_counts(normalize=True).rename("proportion"))
