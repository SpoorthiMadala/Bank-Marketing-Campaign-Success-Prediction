# Bank-Marketing-Campaign-Success-Prediction

## Overview
This project predicts whether a customer will subscribe to a term deposit after a bank marketing campaign. The dataset is from the **UCI Machine Learning Repository** and includes customer information, campaign details, and economic indicators.

## Dataset
The dataset contains 21 attributes, including:
- **Demographic attributes:** age, job, marital status, education
- **Financial attributes:** loan, housing
- **Campaign-related attributes:** contact type, duration, previous campaign outcome
- **Economic indicators:** employment variation rate, consumer confidence index
- **Target variable:** `y` (binary classification: "yes" or "no" for subscription)

## Project Workflow
1. **Data Preprocessing**
   - Load dataset from **UCI Machine Learning Repository**.
   - Convert categorical variables using `OneHotEncoder` or `LabelEncoder`.
   - Standardize numerical features using `StandardScaler`.

2. **Model Training & Hyperparameter Tuning**
   - Train a **Random Forest Classifier** using `GridSearchCV` for optimization.
   - Train an **XGBoost Classifier**.
   - Evaluate models using accuracy, classification report, and confusion matrix.

3. **Cross-Validation & Evaluation**
   - Apply **Stratified K-Fold Cross-Validation** to ensure robustness.
   - Compare models with **cross-validation accuracy**.

## Installation
To run this project, install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost requests
