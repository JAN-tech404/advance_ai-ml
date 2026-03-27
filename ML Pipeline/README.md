# Telco Customer Churn Pipeline

ML pipeline for predicting customer churn using the IBM Telco Customer Churn dataset.

## Results

- **Test Accuracy**: 80.55%
- **Best Model**: Logistic Regression (C=10.0, solver=lbfgs)

## Quick Start

```python
import joblib

# Load the saved pipeline
pipeline = joblib.load("telco_churn_pipeline.pkl")

# Predict churn for a customer
prediction = pipeline.predict([customer_data])
churn_probability = pipeline.predict_proba([customer_data])
```

## Training

Run `ml_pipeline.ipynb` to train the model yourself. The notebook includes:

1. Data loading and preprocessing
2. Feature engineering
3. Pipeline creation with preprocessing + classifier
4. Grid search over Logistic Regression and Random Forest
5. Model evaluation and saving

## Dataset

[IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

## Requirements

```bash
pip install pandas numpy scikit-learn joblib
```

## Files

- `ml_pipeline.ipynb` - Training notebook with GridSearchCV
- `telco_churn_pipeline.pkl` - Saved sklearn pipeline (preprocessing + model)
