# 📊 Customer Churn Analytics

An interactive Machine Learning application that predicts telecom customer churn using the **K-Nearest Neighbors (KNN)** algorithm. Built with Python, Scikit-learn and Streamlit, this project helps identify customers who are likely to discontinue their subscriptions.

## 🚀 Live Application

**[Launch Customer Churn Analytics](https://customer-churn-analytics-k-nearest-neighbors.streamlit.app/)**

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. Losing customers can negatively impact revenue and long-term business growth.

The objective of this project is to develop a machine learning model that predicts whether a telecom customer is likely to churn based on their demographics, subscription details, service usage and billing information.

The application provides an interactive dashboard where users can enter customer details and receive a churn prediction along with the estimated probability.

## ✨ Key Features

- **Customer Churn Prediction:** Predict whether a customer is likely to leave the service.
- **Probability Estimation:** Display estimated churn and retention probabilities.
- **Interactive Dashboard:** Clean and professional Streamlit interface.
- **Model Performance:** Explore classification metrics and model evaluation results.
- **Threshold Optimization:** Uses a classification threshold of 0.40 to improve churn detection.

## 🧠 Machine Learning Approach

**Algorithm:** K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm that classifies a new observation based on the majority class among its nearest neighboring observations.

### Workflow

1. Data loading and exploratory data analysis.
2. Data cleaning and preprocessing.
3. Feature selection and feature engineering.
4. Categorical feature encoding using Ordinal Encoding.
5. Numerical feature scaling using StandardScaler.
6. Model training using K-Nearest Neighbors.
7. Model evaluation and classification threshold tuning.
8. Model serialization using Joblib.
9. Deployment using Streamlit Community Cloud.

## 📈 Model Performance

The model was evaluated using a test dataset of 1,409 observations.

| Evaluation Metric | Result |
|---|---:|
| ROC-AUC Score | 79.03% |
| Churn Precision (0.40 threshold) | 49% |
| Churn Recall (0.40 threshold) | 74% |
| Churn F1-Score (0.40 threshold) | 59% |

### Classification Threshold Optimization

The classification threshold was adjusted from 0.50 to 0.40 to improve the model's ability to identify customers likely to churn.

At a threshold of 0.40:
- Churn recall increased from approximately 51% to 74%.
- The model identifies a larger proportion of actual churn customers.
- This comes with a trade-off in precision and overall accuracy.

**Confusion Matrix at 0.40 Threshold**

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 745 | 291 |
| Actual Churn | 97 | 276 |

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Version Control | Git, GitHub |

## 📂 Project Structure

```text
Customer-Churn-Analytics/
│
├── app.py
├── knn_model.pkl
├── scaler.pkl
├── ordinal_encoder.pkl
├── requirements.txt
└── README.md
```

## 💻 Run Locally

Follow these steps to run the application on your local machine.

**1. Clone the repository**

```bash
git clone https://github.com/tusharr29/Customer-Churn-Analytics.git
```

**2. Navigate to the project directory**

```bash
cd Customer-Churn-Analytics
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔮 Future Improvements

- Compare KNN with Logistic Regression, Decision Trees and Random Forest.
- Perform hyperparameter optimization using GridSearchCV.
- Improve model performance through feature selection.
- Add explainability techniques to understand individual predictions.
- Explore customer segmentation using K-Means clustering.
