# Stroke Prediction Project

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Project Structure](#2-project-structure)
- [3. Dataset Description](#3-dataset-description)
- [4. Methodology](#4-methodology)
  - [4.1 Import Libraries & Load Data](#41-import-libraries--load-data)
  - [4.2 Data Cleaning & Type Conversion](#42-data-cleaning--type-conversion)
  - [4.3 Exploratory Data Analysis (EDA)](#43-exploratory-data-analysis-eda)
  - [4.4 Feature Engineering](#44-feature-engineering)
  - [4.5 Data Preprocessing Pipeline](#45-data-preprocessing-pipeline)
- [5. Model Training & Comparison](#5-model-training--comparison)
- [6. Final Model](#6-final-model)
- [7. How to Run](#7-how-to-run)
- [8. Requirements](#8-requirements)
- [9. Future Improvements](#9-future-improvements)

## 1. Introduction
This project develops a machine learning model for predicting stroke risk using demographic and health-related features. The workflow includes data cleaning, EDA, feature engineering, preprocessing, model training, comparison, and selection of the best model.

## 2. Project Structure
Project layout:
Troke_Prediction.ipynb — main notebook
data/stroke.csv — dataset (if included)
models/ — saved models (optional)
README.md — documentation
requirements.txt — environment setup

## 3. Dataset Description
Features include gender, age, hypertension, heart disease, marital status, work type, residence type, glucose level, BMI, smoking status, and the target variable (stroke: 0/1).

## 4. Methodology

### 4.1 Import Libraries & Load Data
Using pandas, numpy, matplotlib, seaborn, scikit-learn, optionally XGBoost/LightGBM.

### 4.2 Data Cleaning & Type Conversion
Convert categorical fields, handle missing BMI values, ensure correct types, detect anomalies.

### 4.3 Exploratory Data Analysis (EDA)
Analyze distributions of BMI and glucose, stroke percentages across demographic categories, correlation matrix, and outlier detection.

### 4.4 Feature Engineering
Create BMI groups, encode categorical features, impute missing values, scale numerical features.

### 4.5 Data Preprocessing Pipeline
Use ColumnTransformer with:
SimpleImputer for missing values
StandardScaler for numerical features
OneHotEncoder for categorical features
Ensures clean, reproducible ML pipeline.

## 5. Model Training & Comparison
Models evaluated: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVC, KNN.
Metrics used: accuracy, precision, recall, F1-score, ROC-AUC.
A comparison table identifies the best-performing model.

## 6. Final Model
Best model selected based on ROC-AUC, balanced precision/recall, and validation stability. Notebook includes final inference examples.

## 7. How to Run
Create environment:
python -m venv venv
source venv/bin/activate   (macOS/Linux)
venv\Scripts\activate      (Windows)

Install dependencies:
pip install -r requirements.txt

Run notebook:
jupyter notebook Troke_Prediction.ipynb

## 8. Requirements
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost

## 9. Future Improvements
Address class imbalance with SMOTE or class weights, perform hyperparameter tuning (Optuna/GridSearch), deploy model via FastAPI, build interactive dashboard with Streamlit.
