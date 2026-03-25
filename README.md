# 🚀 Customer Churn Prediction API (FastAPI + XGBoost)

## 📌 Overview

This project is an **end-to-end Machine Learning system** that predicts whether a customer will churn or not.
It uses **XGBoost** for modeling and is deployed using **FastAPI** to provide real-time predictions via a REST API.

---

## 🎯 Problem Statement

Customer churn is a major concern for businesses. Retaining existing customers is more cost-effective than acquiring new ones.

👉 The goal of this project is to:

* Predict whether a customer will leave (churn)
* Help businesses take proactive actions to retain customers

---

## 📊 Dataset

* Source: Kaggle (Bank Customer Churn Dataset)
* Target Variable: `Exited`

  * `0` → Not Churn
  * `1` → Churn

### Features:

* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

---

## ⚙️ Approach

### 1. Data Preprocessing

* Removed unnecessary columns (`RowNumber`, `CustomerId`, `Surname`)
* Handled categorical variables using **One-Hot Encoding**
* Feature scaling using **StandardScaler**

---

### 2. Exploratory Data Analysis (EDA)

* Analyzed churn distribution
* Identified important features like:

  * Age
  * Balance
  * Activity status

---

### 3. Model Training

Trained multiple models:

* Logistic Regression
* Random Forest
* **XGBoost (Final Model)**

---

### 4. Model Evaluation

| Metric            | Score    |
| ----------------- | -------- |
| Accuracy          | 82%      |
| Recall (Churn)    | 68%      |
| Precision (Churn) | 56%      |
| ROC-AUC           | **0.85** |

👉 Focused on **Recall** to minimize missed churn customers.

---

### 5. Model Optimization

* Handled class imbalance using `class weight=balanced`
* Tuned hyperparameters
* Adjusted prediction threshold for better recall

---

## 🚀 Deployment

The model is deployed using **FastAPI** and hosted on **Render** for real-time inference.

🔗 **Live Demo:**
👉 https://cust-churn-prediction.onrender.com
👉 https://cust-churn-prediction.onrender.com/docs

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* FastAPI
* Uvicorn
* Render (Deployment)

---

## 📡 API Endpoints

### 🔹 Home

```
GET /
```

Response:

```json
{
  "message": "Churn Prediction API is running 🚀"
}
```

---

### 🔹 Predict Churn

```
POST /predict
```

### Sample Input:

```json
{
  "CreditScore": 600,
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000,
  "Geography_France": 1,
  "Geography_Germany": 0,
  "Geography_Spain": 0,
  "Gender_Female": 0,
  "Gender_Male": 1
}
```

### Sample Output:

```json
{
  "prediction": 0,
  "label": "Not Churn",
  "churn_probability": 0.14
}
```

---

## 📂 Project Structure

```
churn-project/
│
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## ⚙️ Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/ashishjadhav-001/Cust-Churn-Prediction
cd Cust-Churn-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run API locally

```bash
uvicorn app:app --reload
```

### 4. Open in browser

```
http://127.0.0.1:8000/docs
```

---

## 💡 Key Highlights

* End-to-end ML pipeline
* Model comparison and optimization
* Real-time prediction using FastAPI
* Deployed on Render (cloud)
* Production-ready API design

---

## 💼 Resume Description

> Developed an end-to-end Customer Churn Prediction system using XGBoost achieving ROC-AUC of 0.85. Implemented feature engineering, handled class imbalance, and deployed a FastAPI-based REST API on Render for real-time predictions.

---

## 🚀 Future Improvements

* Add frontend (React / Streamlit)
* Deploy on AWS (EC2 / S3)
* Add model monitoring & logging
* Convert to full ML pipeline

---

## 👨‍💻 Author

**Ashish Jadhav**

---

⭐ If you found this project useful, consider giving it a star!
