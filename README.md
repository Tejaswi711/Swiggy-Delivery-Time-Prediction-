#  Swiggy Delivery Time Prediction


A Machine Learning project that predicts food delivery time in minutes using order, rider, location, weather, traffic, vehicle, and time-related features.

## Project Overview

This project builds a regression model to estimate delivery time for food orders. The workflow includes data preprocessing, exploratory data analysis, feature engineering, model comparison, cross-validation, bias-variance analysis, Optuna hyperparameter tuning, feature importance analysis, and model deployment.

## Models Compared

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- KNN Regressor
- XGBoost Regressor

The final model is a tuned XGBoost Regressor.

## Machine Learning Workflow

1. Data loading and understanding
2. Data cleaning and missing-value handling
3. Exploratory Data Analysis
4. Feature engineering
5. Train-test split
6. Preprocessing using Scikit-learn Pipeline and ColumnTransformer
7. Model comparison
8. 5-fold cross-validation
9. Bias-variance / overfitting analysis
10. XGBoost hyperparameter tuning using Optuna
11. Final model training and evaluation
12. Feature importance analysis
13. Model serialization using Pickle
14. Web application deployment

## Web Application

The project includes a Flask web application (`app.py`) that loads the trained pipeline and predicts delivery time from user inputs.

## Project Structure

```text
Swiggy-Delivery-Time-Prediction/
├── app.py
├── swiggy_xgb_pipeline.pkl
├── Swiggy_Delivery_Time_Prediction_XGBoost.ipynb
├── requirements.txt
└── README.md
