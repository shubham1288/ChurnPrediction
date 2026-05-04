# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project to predict customer churn for a fictional telecom company. This project covers the full ML lifecycle — from data preprocessing to model building and evaluation.

---

## 🚀 Project Overview

Customer churn is a critical business problem in the telecom industry. This project aims to identify customers who are likely to leave using historical data and predictive modeling techniques.

---

## 📂 Dataset Description

The dataset contains **5,000 customer records** with demographic, account, and service-related information.

### 🧑‍🤝‍🧑 Customer Demographics
- `customer_id` — Unique identifier
- `age` — Customer age (18–75)
- `gender` — Male / Female / Other
- `region` — North / South / East / West
- `senior_citizen` — 0 or 1
- `partner` — 0 or 1

### 📞 Account & Usage
- `tenure_months` — Months as customer
- `contract_type` — Monthly / 1 Year / 2 Year
- `monthly_charges` — Monthly bill (₹)
- `total_charges` — Lifetime charges (₹)
- `avg_monthly_calls` — Call volume
- `data_usage_gb` — Monthly data usage
- `support_tickets` — Tickets raised
- `payment_method` — Payment method

### 📡 Services & Target
- `internet_service` — DSL / Fiber / None
- `streaming_tv` — 0 or 1
- `tech_support` — Yes / No
- `online_backup` — Yes / No
- `churn` — Target variable (0 = Stay, 1 = Churn)

---

## ⚠️ Data Challenges

- ~4% missing values in `total_charges`
- ~2% missing values in `data_usage_gb`
- Outliers in `monthly_charges`
- Class imbalance (~22% churn rate)
- Duplicate `customer_id` entries (~15 rows)

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost / LightGBM (optional)

---

## 🔄 Project Pipeline

### 1. Data Preprocessing
- Handle missing values
- Remove duplicates
- Treat outliers
- Encode categorical variables
- Scale features

### 2. Exploratory Data Analysis (EDA)
- Analyze churn distribution
- Identify correlations
- Visualize patterns

### 3. Feature Engineering
- Create derived features
- Transform variables

### 4. Handling Class Imbalance
- SMOTE / undersampling / class weights

### 5. Model Building
- Logistic Regression
- Random Forest
- Gradient Boosting

### 6. Model Evaluation
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC

---

## 📈 Results

Key churn indicators:
- High monthly charges
- Low tenure
- High support tickets
- Lack of tech support

---

## 📦 Live Model
https://shubham128-churnprediction.hf.space/
