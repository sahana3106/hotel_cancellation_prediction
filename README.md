# 🏨 Hotel Cancellation Prediction & Analytics

An end-to-end Machine Learning project that analyzes hotel booking patterns and predicts whether a hotel booking is likely to be cancelled.

The project combines **Exploratory Data Analysis, Feature Engineering, Machine Learning, and an interactive Streamlit dashboard** to provide both analytical insights and booking-level cancellation predictions.

---

## 🚀 Live Demo

🔗 **Live Streamlit App:**  
PASTE YOUR STREAMLIT APP LINK HERE

---

## 📌 Project Overview

Hotel booking cancellations can affect hotel revenue, room availability, and operational planning.

This project aims to:

- Analyze hotel booking and cancellation patterns
- Identify factors associated with booking cancellations
- Perform data preprocessing and feature engineering
- Build a machine learning classification model
- Predict whether a booking will be cancelled
- Provide an interactive dashboard for data exploration
- Deploy the application using Streamlit Community Cloud

---

## 📊 Dataset

The project uses a hotel booking dataset containing **119,390 bookings**.

The dataset contains information related to:

- Hotel type
- Lead time
- Arrival dates
- Length of stay
- Number of guests
- Previous cancellations
- Booking changes
- Deposit type
- Customer information
- Special requests
- Average Daily Rate (ADR)
- Cancellation status

### Target Variable

**`is_canceled`**

- `0` → Booking was not cancelled
- `1` → Booking was cancelled

---

## 🔎 Exploratory Data Analysis

The dataset was explored using Python and Pandas to understand booking behaviour and cancellation patterns.

The dashboard provides analysis such as:

- Total bookings
- Cancelled bookings
- Overall cancellation rate
- Cancellation rate by hotel type
- Cancellation rate by deposit type
- Other booking-related patterns

### Key Dataset Statistics

| Metric | Value |
|---|---:|
| Total Bookings | 119,390 |
| Cancelled Bookings | 44,224 |
| Cancellation Rate | 37.0% |

---

## ⚙️ Feature Engineering

Additional features were created to improve the representation of booking behaviour.

Examples include:

- `total_night_stay`
- `total_guest`
- `has_company`

The preprocessing pipeline also handles:

### Numerical Features

- Missing value imputation
- Standardization using `StandardScaler`

### Categorical Features

- Missing value imputation
- Conversion to string format
- One-hot encoding using `OneHotEncoder`
- Handling previously unseen categories

---

## 🤖 Machine Learning Model

The project uses a:

### Random Forest Classifier

The final model is implemented as a Scikit-learn Pipeline:

```text
Input Data
    ↓
ColumnTransformer
    ├── Numerical Features
    │      ↓
    │   SimpleImputer
    │      ↓
    │   StandardScaler
    │
    └── Categorical Features
           ↓
        SimpleImputer
           ↓
        String Conversion
           ↓
        OneHotEncoder
    ↓
Random Forest Classifier
