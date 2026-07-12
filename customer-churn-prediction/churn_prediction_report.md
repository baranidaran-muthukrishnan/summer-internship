# Customer Churn Prediction

## Executive summary

This project predicts whether a customer is likely to leave a service using historical behavior, billing, support, contract, and satisfaction signals. The generated historical dataset contains 5,000 customers with an observed churn rate of 26.3%.

The strongest model in this run is **Logistic regression**, with ROC-AUC of **0.810** and F1 score of **0.508** on a held-out 25% test set.

## Model comparison

| model               | accuracy | precision | recall | f1    | roc_auc |
| ------------------- | -------- | --------- | ------ | ----- | ------- |
| Logistic regression | 0.774    | 0.661     | 0.412  | 0.508 | 0.810   |
| Decision tree       | 0.753    | 0.611     | 0.350  | 0.445 | 0.740   |

## Key churn factors

- contract_Two year: lowers churn risk (standardized impact 0.706)
- contract_One year: lowers churn risk (standardized impact 0.516)
- tenure_months: lowers churn risk (standardized impact 0.505)
- satisfaction_score: lowers churn risk (standardized impact 0.440)
- monthly_charges: increases churn risk (standardized impact 0.386)
- internet_service_Fiber optic: increases churn risk (standardized impact 0.305)
- payment_method_Electronic check: increases churn risk (standardized impact 0.292)
- total_charges: increases churn risk (standardized impact 0.257)

## Churn by contract

- Month-to-month: 36.6%
- One year: 16.0%
- Two year: 7.4%

## Risk concentration

The model is useful for retention targeting because churn concentrates in the highest-risk groups:

| risk_group | customers | churn_rate | avg_probability |
| ---------- | --------- | ---------- | --------------- |
| Top 10%    | 125       | 0.736      | 0.744           |
| Top 20%    | 125       | 0.568      | 0.543           |
| Top 30%    | 125       | 0.392      | 0.409           |
| Top 40%    | 125       | 0.376      | 0.305           |
| Top 50%    | 125       | 0.296      | 0.229           |

## Recommended retention actions

- Prioritize customers with month-to-month contracts, high monthly charges, low satisfaction, recent support tickets, late payments, or electronic-check payments.
- Offer contract migration incentives to high-risk month-to-month customers.
- Trigger service recovery workflows after repeated support tickets or outages.
- Promote auto-pay and lower-friction payment methods for customers with late-payment history.
- Use the prediction file to build a weekly outreach queue sorted by predicted churn probability.

## Generated files

- `customer_churn_historical_data.csv`: synthetic historical customer records used for training.
- `customer_churn_predictions.csv`: held-out customer predictions ranked by churn probability.
- `model_metrics.csv`: classification metrics for both models.
- `key_churn_factors.csv`: model coefficients and feature impact ranking.
- `risk_decile_lift.csv`: churn concentration by predicted-risk decile.
- `key_churn_factors.svg`: feature impact chart.
- `churn_rate_by_contract.svg`: churn-rate chart by contract type.
