"""
Parkinson's Disease Detection - Model Training Script
Achieves 90+% accuracy using ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings

warnings.filterwarnings("ignore")

# Load UCI Parkinson's Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nClass distribution:")
print(df["status"].value_counts())

# Separate features and target
X = df.drop(["name", "status"], axis=1)
y = df["status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("TRAINING MULTIPLE MODELS")
print("=" * 60)

# Model 1: Random Forest with optimized parameters
print("\n1. Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# Model 2: Gradient Boosting
print("\n2. Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    random_state=42,
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Accuracy: {gb_acc*100:.2f}%")

# Model 3: SVM with RBF kernel
print("\n3. Training SVM...")
svm_model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_acc*100:.2f}%")

# Ensemble Model: Voting Classifier
print("\n4. Creating Ensemble Model...")
ensemble_model = VotingClassifier(
    estimators=[("rf", rf_model), ("gb", gb_model), ("svm", svm_model)],
    voting="soft",
    weights=[2, 2, 1],  # Give more weight to RF and GB
)
ensemble_model.fit(X_train_scaled, y_train)
ensemble_pred = ensemble_model.predict(X_test_scaled)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Random Forest:      {rf_acc*100:.2f}%")
print(f"Gradient Boosting:  {gb_acc*100:.2f}%")
print(f"SVM:                {svm_acc*100:.2f}%")
print(f"ENSEMBLE MODEL:     {ensemble_acc*100:.2f}%")

# Detailed evaluation of ensemble model
print("\n" + "=" * 60)
print("ENSEMBLE MODEL DETAILED EVALUATION")
print("=" * 60)
print("\nClassification Report:")
print(
    classification_report(
        y_test, ensemble_pred, target_names=["Healthy", "Parkinson's"]
    )
)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, ensemble_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives:  {cm[1][1]}")

# Cross-validation score
cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Feature importance from Random Forest
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 60)
print(feature_importance.head(10).to_string(index=False))

# Save the model and scaler
print("\n" + "=" * 60)
print("SAVING MODEL AND SCALER")
print("=" * 60)

with open("parkinsons_model.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)
print("Model saved as 'parkinsons_model.pkl'")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'scaler.pkl'")

# Save feature names for reference
with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
print("Feature names saved as 'feature_names.pkl'")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nFinal Model Accuracy: {ensemble_acc*100:.2f}%")
print("\nFiles created:")
print("  - parkinsons_model.pkl")
print("  - scaler.pkl")
print("  - feature_names.pkl")
print("\nYou can now use these files in the web application!")
