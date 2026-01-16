# CreditWise Loan Approval System - Project Overview

## 📋 Executive Summary

This project implements an **Intelligent Loan Approval System** for **SecureTrust Bank** using Machine Learning. The system analyzes historical loan application data to predict whether a loan should be **Approved** or **Rejected**, replacing the traditional manual verification process that was time-consuming, biased, and inconsistent.

---

## 🎯 Problem Statement

SecureTrust Bank faced two major challenges with their manual loan approval process:

1. **Loss of Business**: Good customers sometimes get rejected
2. **Financial Losses**: High-risk customers sometimes get approved

The goal is to create an ML-powered system that provides **accurate, fast, and unbiased** loan approval decisions.

---

## 📊 Dataset Description

**File**: `2 loan_approval_data.csv`

### Dataset Features (20 columns)

| Feature              | Type        | Description                                           |
| -------------------- | ----------- | ----------------------------------------------------- |
| `Applicant_ID`       | Numeric     | Unique applicant identifier (dropped during modeling) |
| `Applicant_Income`   | Numeric     | Monthly income of applicant                           |
| `Coapplicant_Income` | Numeric     | Monthly income of co-applicant                        |
| `Employment_Status`  | Categorical | Salaried / Self-Employed / Contract / Unemployed      |
| `Age`                | Numeric     | Applicant's age                                       |
| `Marital_Status`     | Categorical | Married / Single                                      |
| `Dependents`         | Numeric     | Number of dependents                                  |
| `Credit_Score`       | Numeric     | Credit bureau score                                   |
| `Existing_Loans`     | Numeric     | Number of active loans                                |
| `DTI_Ratio`          | Numeric     | Debt-to-Income ratio                                  |
| `Savings`            | Numeric     | Savings balance                                       |
| `Collateral_Value`   | Numeric     | Value of collateral provided                          |
| `Loan_Amount`        | Numeric     | Loan amount requested                                 |
| `Loan_Term`          | Numeric     | Loan duration in months                               |
| `Loan_Purpose`       | Categorical | Home / Education / Personal / Business / Car          |
| `Property_Area`      | Categorical | Urban / Semi-Urban / Rural                            |
| `Education_Level`    | Categorical | Graduate / Postgraduate / Not Graduate                |
| `Gender`             | Categorical | Male / Female                                         |
| `Employer_Category`  | Categorical | Govt / Private / MNC / Business / Unemployed          |
| **`Loan_Approved`**  | **Target**  | **1 = Approved, 0 = Rejected**                        |

---

## 🔧 Data Preprocessing Pipeline

### 1. Missing Value Handling

- **Numerical Features**: Imputed using **mean** strategy
- **Categorical Features**: Imputed using **most_frequent** (mode) strategy

### 2. Feature Encoding

- **Label Encoding**: Applied to `Education_Level` and `Loan_Approved`
- **One-Hot Encoding**: Applied to categorical features with `drop='first'` to avoid multicollinearity:
  - `Employment_Status`
  - `Marital_Status`
  - `Loan_Purpose`
  - `Property_Area`
  - `Gender`
  - `Employer_Category`

### 3. Feature Engineering (Enhanced Models)

Three new features were created to improve model performance:

- **`DTI_Ratio_sq`**: Squared DTI Ratio (capturing non-linear relationships)
- **`Credit_Score_sq`**: Squared Credit Score
- **`Applicant_Income_log`**: Log-transformed Applicant Income (normalizing skewed distribution)

### 4. Feature Scaling

- **StandardScaler** applied for normalization (mean=0, std=1)

### 5. Train-Test Split

- **Test Size**: 20%
- **Random State**: 42 (for reproducibility)

---

## 📈 Exploratory Data Analysis (EDA)

### Key Visualizations Performed:

1. **Class Distribution (Pie Chart)**: Analyzing loan approval balance
2. **Gender Distribution (Bar Plot)**: Understanding demographic distribution
3. **Education Level Distribution**: Analyzing education patterns
4. **Income Distribution (Histograms)**: Applicant and Co-applicant income analysis with KDE
5. **Box Plots**: Outlier detection for:
   - Applicant Income vs Loan Status
   - Credit Score vs Loan Status
   - DTI Ratio vs Loan Status
   - Savings vs Loan Status
6. **Credit Score vs Loan Approval (Stacked Histogram)**: Key relationship analysis
7. **Correlation Heatmap**: Understanding feature relationships

### Key Insights from Correlation Analysis:

The correlation analysis reveals the relationship between features and loan approval, helping identify the most predictive features.

---

## 🤖 Machine Learning Models

### Models Implemented:

1. **Logistic Regression** - Linear classification baseline
2. **K-Nearest Neighbors (KNN)** - Instance-based learning (k=7)
3. **Gaussian Naive Bayes** - Probabilistic classifier

---

## 📊 Model Comparison

### Before Feature Engineering

| Model                   | Precision | Accuracy  | Recall    | F1 Score  |
| ----------------------- | --------- | --------- | --------- | --------- |
| **Logistic Regression** | 0.785     | **0.880** | **0.836** | **0.810** |
| **Naive Bayes**         | **0.804** | 0.865     | 0.738     | 0.769     |
| KNN (k=7)               | 0.609     | 0.745     | 0.459     | 0.523     |

### After Feature Engineering

| Model                   | Precision | Accuracy  | Recall    | F1 Score  |
| ----------------------- | --------- | --------- | --------- | --------- |
| **Logistic Regression** | 0.785     | **0.880** | **0.836** | **0.810** |
| **Naive Bayes**         | **0.811** | 0.860     | 0.705     | 0.754     |
| KNN (k=7)               | 0.646     | 0.765     | 0.508     | 0.569     |

---

## 📈 Detailed Model Analysis

### 1. Logistic Regression 🏆 **Best Overall Performance**

| Metric        | Value | Interpretation                                       |
| ------------- | ----- | ---------------------------------------------------- |
| **Accuracy**  | 88.0% | Correctly classified 88% of all applications         |
| **Precision** | 78.5% | Of predicted approvals, 78.5% were actually approved |
| **Recall**    | 83.6% | Caught 83.6% of actual approved loans                |
| **F1 Score**  | 81.0% | Best balance between precision and recall            |

**Confusion Matrix:**

```
              Predicted
            Reject  Approve
Actual  Reject  125      14
        Approve  10      51
```

**Strengths:**

- Highest accuracy (88%)
- Best F1 Score (81%)
- Best recall - minimizes missed good customers
- Good interpretability

**Weaknesses:**

- Slightly lower precision than Naive Bayes

---

### 2. Gaussian Naive Bayes 🥈 **Best Precision**

| Metric        | Value     | Interpretation                                       |
| ------------- | --------- | ---------------------------------------------------- |
| **Accuracy**  | 86.5%     | Correctly classified 86.5% of all applications       |
| **Precision** | **80.4%** | Highest - Of predicted approvals, 80.4% were correct |
| **Recall**    | 73.8%     | Caught 73.8% of actual approved loans                |
| **F1 Score**  | 76.9%     | Good balance                                         |

**Confusion Matrix:**

```
              Predicted
            Reject  Approve
Actual  Reject  128      11
        Approve  16      45
```

**Strengths:**

- Highest precision - minimizes approving bad customers
- Fast training and prediction
- Works well with small datasets

**Weaknesses:**

- Lower recall - misses more good customers
- Assumes feature independence

---

### 3. K-Nearest Neighbors (k=7) 🥉

| Metric        | Value | Interpretation                                 |
| ------------- | ----- | ---------------------------------------------- |
| **Accuracy**  | 74.5% | Correctly classified 74.5% of all applications |
| **Precision** | 60.9% | Of predicted approvals, 60.9% were correct     |
| **Recall**    | 45.9% | Only caught 45.9% of actual approved loans     |
| **F1 Score**  | 52.3% | Lowest performance                             |

**Confusion Matrix:**

```
              Predicted
            Reject  Approve
Actual  Reject  121      18
        Approve  33      28
```

**Strengths:**

- Non-parametric approach
- Can capture complex patterns

**Weaknesses:**

- Lowest performance across all metrics
- Sensitive to feature scaling
- May need hyperparameter tuning

---

## 📊 Visual Model Comparison

### Performance Metrics Chart

```
Accuracy (%)
═══════════════════════════════════════════════════════
Logistic Regression  ████████████████████████████████ 88.0%
Naive Bayes          ██████████████████████████████   86.5%
KNN (k=7)            ████████████████████████         74.5%

Precision (%)
═══════════════════════════════════════════════════════
Naive Bayes          ████████████████████████████████ 80.4%
Logistic Regression  ███████████████████████████████  78.5%
KNN (k=7)            ████████████████████████         60.9%

Recall (%)
═══════════════════════════════════════════════════════
Logistic Regression  ████████████████████████████████ 83.6%
Naive Bayes          █████████████████████████████    73.8%
KNN (k=7)            ██████████████████               45.9%

F1 Score (%)
═══════════════════════════════════════════════════════
Logistic Regression  ████████████████████████████████ 81.0%
Naive Bayes          ██████████████████████████████   76.9%
KNN (k=7)            █████████████████████            52.3%
```

---

## 🏆 Model Selection Recommendation

### Primary Recommendation: **Logistic Regression**

| Criteria                     | Winner              | Reason                                  |
| ---------------------------- | ------------------- | --------------------------------------- |
| **Overall Accuracy**         | Logistic Regression | 88% vs 86.5% (NB)                       |
| **Balanced Performance**     | Logistic Regression | Best F1 Score (81%)                     |
| **Minimize False Negatives** | Logistic Regression | Highest Recall (83.6%)                  |
| **Minimize False Positives** | Naive Bayes         | Highest Precision (80.4%)               |
| **Interpretability**         | Logistic Regression | Coefficients explain feature importance |

### Business Context Consideration:

- **If minimizing financial risk is priority** → Use **Naive Bayes** (higher precision)
- **If maximizing customer acquisition is priority** → Use **Logistic Regression** (higher recall)
- **For balanced approach** → Use **Logistic Regression** (best F1 Score)

---

## 🛠️ Technology Stack

| Component                   | Technology                 |
| --------------------------- | -------------------------- |
| **Language**                | Python 3.x                 |
| **Data Processing**         | Pandas, NumPy              |
| **Visualization**           | Matplotlib, Seaborn        |
| **Machine Learning**        | Scikit-learn               |
| **Development Environment** | Jupyter Notebook (VS Code) |

### Libraries Used:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
```

---

## 📁 Project Structure

```
ML_projects/
├── 2 loan_approval_data.csv    # Dataset file
├── credit_wise.ipynb           # Main Jupyter Notebook
├── Project_probelm.md          # Problem statement document
└── PROJECT_OVERVIEW.md         # This comprehensive overview
```

---

## 🔮 Future Improvements

1. **Hyperparameter Tuning**: Use GridSearchCV/RandomSearchCV for optimal parameters
2. **Additional Models**: Test Random Forest, XGBoost, SVM
3. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
4. **Feature Selection**: Use techniques like RFE, feature importance
5. **Handle Class Imbalance**: Apply SMOTE or class weights if needed
6. **Model Deployment**: Create REST API using Flask/FastAPI
7. **Explainability**: Add SHAP values for model interpretation

---

## 📝 Conclusion

The **Logistic Regression** model emerges as the best choice for the CreditWise Loan Approval System with:

- **88% accuracy**
- **81% F1 Score**
- Excellent balance between precision and recall
- Good interpretability for regulatory compliance

This ML-powered system will help SecureTrust Bank make **faster, more accurate, and unbiased** loan approval decisions, addressing both the risk of approving bad customers and the opportunity cost of rejecting good ones.

---

_Document Generated: January 2026_  
_Project: CreditWise Loan Approval System_  
_Author: ML Engineering Team_
# CreditWise-Loan-Approval-System---Project-Overview
