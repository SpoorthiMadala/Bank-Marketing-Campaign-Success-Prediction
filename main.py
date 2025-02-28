import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import io
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open("bank-additional/bank-additional-full.csv") as f:
        data = pd.read_csv(f, sep=';')

X = data.drop(columns=['y'])
y = data['y']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5], 'max_features': ['sqrt']}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)
best_rf_model = grid_search.best_estimator_

y_pred_rf = best_rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nOptimized Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title("Confusion Matrix Heatmap (Random Forest)")
plt.show()

xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nXGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%")

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_rf = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=stratified_kfold, n_jobs=-1)
cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=stratified_kfold, n_jobs=-1)

print("\nStratified K-Fold CV Accuracy Scores (Random Forest):", cv_scores_rf)
print(f"Mean Accuracy: {cv_scores_rf.mean() * 100:.2f}%")

print("\nStratified K-Fold CV Accuracy Scores (XGBoost):", cv_scores_xgb)
print(f"Mean Accuracy: {cv_scores_xgb.mean() * 100:.2f}%")

plt.figure(figsize=(8, 6))
plt.bar(['Random Forest'], cv_scores_rf.mean(), alpha=0.7)
plt.bar(['XGBoost'], cv_scores_xgb.mean(), alpha=0.7)
plt.ylabel('Average Accuracy')
plt.title('Stratified K-Fold Model Comparison')
plt.show()
