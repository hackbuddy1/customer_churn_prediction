# Customer Churn Prediction

## Project Overview
Built an end-to-end ML pipeline to predict customer churn for a telecom company using real-world data of 7032 customers.

## Problem Statement
Telecom companies lose revenue when customers leave. This project identifies at-risk customers before they churn so the company can take retention action.

## Key Findings (EDA)
- 27% customers churned — imbalanced dataset
- Fiber optic customers showed 42% churn rate
- Month-to-month contract = highest churn risk
- Customers with lower tenure churned more

## Models Used
| Model | Accuracy | Recall (Churn) |
|---|---|---|
| Logistic Regression | 78.75% | 0.51 |
| Random Forest | 79.03% | 0.49 |
| XGBoost + scale_pos_weight | 74.20% | 0.67 |

## Why XGBoost?
Recall was prioritized over accuracy — missing a churning customer costs more than a false alarm.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- Matplotlib, Seaborn

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## App Features
- Upload customer CSV
- Instant churn prediction
- Risk categorization — High/Medium/Safe
- Download results
