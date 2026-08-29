# Customer Churn Prediction 📉

A complete, end-to-end machine learning project that predicts whether a telecom customer is likely to **churn** (cancel their subscription), using customer account, service, and billing data.

This project covers the full ML workflow: data generation/loading → exploratory data analysis → preprocessing → model training & comparison → evaluation → inference on new data.

---

## 🎯 Problem Statement

Customer churn is one of the most expensive problems in subscription-based businesses — acquiring a new customer typically costs far more than retaining an existing one. This project builds a binary classification model that flags customers at high risk of churning, so a business could proactively target them with retention offers.

**Target variable:** `churn` (`Yes` / `No`)

---

## 🗂️ Project Structure

```
churn-prediction/
├── data/
│   ├── generate_data.py          # generates the synthetic dataset
│   ├── customer_churn.csv        # generated dataset (5,000 rows)
│   └── sample_new_customers.csv  # example unseen data for inference
├── notebooks/
│   └── eda.py                    # exploratory data analysis, saves figures
├── src/
│   ├── preprocess.py             # shared preprocessing pipeline
│   ├── train.py                  # trains & compares models, saves the best
│   └── predict.py                # scores new customers with the saved model
├── models/
│   └── churn_model.joblib        # trained pipeline (preprocessing + model)
├── reports/
│   ├── figures/                  # EDA and evaluation charts
│   ├── model_comparison.csv      # metrics for every model tried
│   └── best_model_metrics.json   # metrics for the selected model
├── tests/
│   └── test_pipeline.py          # unit tests for the pipeline
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

---

## 📊 Dataset

The dataset is **synthetically generated** (`data/generate_data.py`) but modeled on real telecom churn drivers, so the relationships are realistic rather than random:

| Feature | Description |
|---|---|
| `tenure_months` | Months the customer has been with the company |
| `contract` | Month-to-month, One year, or Two year |
| `internet_service` | DSL, Fiber optic, or No internet |
| `monthly_charges` / `total_charges` | Billing amounts |
| `online_security`, `tech_support` | Whether the customer has these add-ons |
| `paperless_billing`, `payment_method` | Billing preferences |
| `num_support_calls` | Number of customer support calls |
| `senior_citizen`, `gender` | Demographics |
| `churn` | **Target** — did the customer churn? |

- **5,000 rows**, no missing values
- Class balance: **75.7% No / 24.3% Yes** (realistic imbalance, similar to industry-reported churn datasets)

Regenerate the dataset any time with:
```bash
python data/generate_data.py
```

---

## 🔍 Exploratory Data Analysis

Run `python notebooks/eda.py` to reproduce the charts in `reports/figures/`. Key findings:

- **Contract type is the strongest churn driver** — month-to-month customers churn at ~36%, vs. ~14% for one-year and ~7% for two-year contracts.
- **New customers churn more** — churn risk drops sharply as tenure increases.
- **Higher monthly charges correlate with higher churn.**
- More **support calls** correlate with higher churn — a signal of dissatisfaction.

![Churn by Contract](reports/figures/churn_by_contract.png)

---

## 🛠️ Methodology

1. **Preprocessing** (`src/preprocess.py`): numeric features are standardized; categorical features are one-hot encoded, all inside a single `sklearn.ColumnTransformer` so training and inference use identical transformations.
2. **Model training** (`src/train.py`): three classifiers are trained and compared using 5-fold cross-validation and a held-out 20% test set:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
3. **Model selection**: the model with the highest test-set ROC-AUC is saved as the final pipeline.
4. **Evaluation**: ROC curves, confusion matrix, and feature importances are generated for the winning model.

---

## 📈 Results

| Model | CV AUC | Test AUC | Precision (Churn) | Recall (Churn) | F1 (Churn) | Accuracy |
|---|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.756 | **0.782** | 0.607 | 0.267 | 0.371 | 0.780 |
| Random Forest | 0.746 | 0.781 | 0.597 | 0.189 | 0.288 | 0.772 |
| Gradient Boosting | 0.744 | 0.772 | 0.545 | 0.247 | 0.340 | 0.767 |

**Logistic Regression** was selected as the final model — it matches the tree-based ensembles on AUC while being simpler and more interpretable, which matters for a business use case like churn (stakeholders can inspect coefficients directly).

![ROC Curves](reports/figures/roc_curves.png)
![Confusion Matrix](reports/figures/confusion_matrix.png)

> **Note on recall:** churn is a minority class (24%), so recall on the churn class is moderate (0.27) at the default 0.5 threshold. In a real deployment, the classification threshold would be tuned based on the business cost of a missed churner vs. the cost of a false alarm (e.g., lowering the threshold to catch more churners at the expense of precision).

---

## 🚀 Getting Started

### 1. Clone and set up the environment
```bash
git clone https://github.com/<your-username>/churn-prediction.git
cd churn-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the data
```bash
python data/generate_data.py
```

### 3. Run EDA
```bash
python notebooks/eda.py
```

### 4. Train and evaluate models
```bash
python src/train.py
```

### 5. Predict on new customers
```bash
python src/predict.py --input data/sample_new_customers.csv --output predictions.csv
```

### 6. Run tests
```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

> All commands above assume you're running from the project root.

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / Optuna) for the ensemble models
- Threshold tuning against a business cost matrix instead of the default 0.5
- SHAP values for per-customer explainability
- A simple Streamlit/Flask app to serve predictions interactively
- Replace synthetic data with a real dataset (e.g., IBM Telco Customer Churn) for a production-grade version

---

## 🧰 Tech Stack

- Python 3.12
- pandas, numpy — data manipulation
- scikit-learn — preprocessing, modeling, evaluation
- matplotlib, seaborn — visualization
- joblib — model persistence
- pytest — testing

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

[Your Name] — built as a complete ML project applying data preprocessing, model comparison, evaluation, and deployment-ready inference.
