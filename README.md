Heart Disease ML Dashboard

An interactive Machine Learning dashboard for predicting the risk of heart disease using patient clinical data.
The application is built with Streamlit and integrates multiple ML models, clustering analysis, and explainable AI.

---

Project Overview

This project implements an end-to-end machine learning pipeline for heart disease prediction including:

- Data preprocessing
- Feature engineering
- Model training and evaluation
- Clustering analysis
- Explainable AI
- Interactive visualization

The dashboard allows users to explore patient data, predict heart disease risk, and understand model predictions.

---

Features

- Real-time heart disease risk prediction
- Ensemble prediction using multiple ML models
- Patient clustering using K-Means and DBSCAN
- Model explainability using SHAP
- Interactive visualization with Plotly
- Batch prediction for multiple patients via CSV upload
- Interactive dashboard using Streamlit

---

Machine Learning Models

The system uses multiple trained models:

- Logistic Regression
- XGBoost
- K-Means Clustering
- DBSCAN Clustering

Predictions are compared and aggregated to provide more reliable risk estimation.

---

Project Structure

├── app.py
├── preprocess.py
├── models/
│   ├── xgb_model.pkl
│   ├── lg_model.pkl
│   ├── kmeans_model.pkl
│   ├── dbscan_model.pkl
│   └── scaler.pkl
├── data/
├── requirements.txt
└── README.md

---

Installation

Clone the repository:

git clone https://github.com/yourusername/heart-disease-dashboard.git
cd heart-disease-dashboard

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

---

Usage

The dashboard provides three main modes:

1. Single Patient Prediction
   
   - Enter patient clinical data
   - Predict heart disease risk in real time

2. Cluster Exploration
   
   - Identify patient clusters based on medical features

3. Batch Prediction
   
   - Upload a CSV file with multiple patient records
   - Download predictions as a CSV file

---

Technologies Used

- Python
- Streamlit
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Pandas
- NumPy

---

Future Improvements

- Add deep learning models
- Deploy the dashboard online
- Add more clinical datasets
- Improve model performance

---
