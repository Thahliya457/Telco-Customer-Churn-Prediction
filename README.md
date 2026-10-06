
# Telco Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to cancel their service, with a Streamlit app for interactive predictions.

![App screenshot 1](app1.png)

## Business Problem

Keeping an existing customer costs less than winning a new one. If a telecom company can spot customers who are likely to leave, it can contact them early with an offer or a fix. This project builds a model that flags those customers.

## Tools

Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Git

## What I Did

- Cleaned the data and encoded categorical variables
- Explored the data (EDA) and prepared features
- Balanced the training data with SMOTE (applied after the train-test split, so the test set is untouched)
- Compared Decision Tree, Random Forest and XGBoost models
- Evaluated the Random Forest model on a held-out test set using a confusion matrix, precision, recall and F1 score
- Built a Streamlit app for churn prediction

## The App

Enter a customer's details (gender, senior citizen status, partner, dependents, tenure, phone and other services) and the app predicts whether the customer is likely to churn.

![App screenshot 2](app2.png)

![App screenshot 3](app3.png)

## Model Evaluation

Models were compared with 5-fold cross-validation on the SMOTE-balanced training data. The selected model was then evaluated on a held-out test set of 1,409 customers.

**Cross-validation accuracy (training data only)**

| Model         | CV accuracy |
| ------------- | ----------- |
| Decision Tree | 0.78        |
| Random Forest | 0.84        |
| XGBoost       | 0.83        |

> These scores are optimistic. SMOTE adds synthetic customers to the data, and some of them land in the validation folds. The test-set results below are the reliable numbers.

**Selected model: Random Forest, test-set results**

| Metric          | Score |
| --------------- | ----- |
| Accuracy        | 0.78  |
| Churn precision | 0.59  |
| Churn recall    | 0.58  |
| Churn F1        | 0.58  |

**Confusion matrix (test set)**

|                | Predicted stay | Predicted churn |
| -------------- | -------------- | --------------- |
| Actually stay  | 885            | 151             |
| Actually churn | 158            | 215             |

## Key Findings

- The model catches **58% of customers who churn** (215 of 373) and is right 59% of the time when it flags someone.
- Always predicting "stays" would score about 73.5% accuracy but catch no churners, so churn recall is a better measure than accuracy here.

## Limitations and Next Steps

- Recall for churners is moderate, so about 4 in 10 churners are missed.
- Only the Random Forest model has been evaluated on the test set so far. Next step: evaluate Decision Tree and XGBoost the same way and confirm the best model.
- Tune the decision threshold and class weights to catch more churners, and tune hyperparameters.
- Deploy the app so it can be tried online.

## Files

- `Customer_Churn_Prediction_using_ML` - notebook with analysis and models
- `app.py` - Streamlit app
- `app1.png`, `app2.png`, `app3.png` - app screenshots
- `customer_churn_model.pkl`, `encoders.pkl` - saved model and encoders
- `requirements.txt` - Python packages needed to run the app
- `WA_Fn-UseC_-Telco-Customer-Churn` - dataset (Telco Customer Churn sample data, available on Kaggle)

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `localhost:8501`. The saved model files were created with specific library versions, so keep the versions listed in `requirements.txt`.

## Author

Thahliya Jasmi | [Portfolio](https://thahliya-portfolio.lovable.app) | [LinkedIn](https://linkedin.com/in/thahliya-jasmi)
