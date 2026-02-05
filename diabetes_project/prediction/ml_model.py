"""
ML training script for Diabetes prediction.
Run this from the project root:

python diabetes_project/prediction/ml_model.py

It will: load dataset, preprocess, train 3 models, evaluate, save best model and scaler,
and export confusion matrix and accuracy comparison plots to static/images.
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_CANDIDATES = [
    BASE_DIR / 'dataset' / 'Diabetes detection.csv',
    BASE_DIR / 'Diabetes detection.csv',
]

DATA_PATH = None
for p in DATA_CANDIDATES:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError('Dataset not found. Place "Diabetes detection.csv" in project root or dataset/ folder')

print(f"Loading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Basic cleaning: replace zeros in certain columns with median (common for Pima dataset)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)
    # Avoid chained assignment / inplace to work with pandas copy-on-write
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    results[name] = {'model': model, 'accuracy': acc, 'confusion_matrix': cm}
    print(f"{name} accuracy: {acc:.4f}")

# Pick best model by accuracy
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
print(f"Best model: {best_name}")

# Save model and scaler
models_dir = BASE_DIR / 'prediction' / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, models_dir / 'best_model.joblib')
joblib.dump(scaler, models_dir / 'scaler.joblib')

# Save confusion matrix for best model
cm = results[best_name]['confusion_matrix']
plt.figure(figsize=(6,4))
# Ensure images directory exists
images_dir = BASE_DIR / 'static' / 'images'
images_dir.mkdir(parents=True, exist_ok=True)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(images_dir / 'confusion_matrix.png')
plt.close()

# Accuracy comparison chart
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
plt.figure(figsize=(6,4))
plt.bar(names, accs, color=['#4c72b0','#dd8452','#55a868'])
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig(BASE_DIR / 'static' / 'images' / 'accuracy_comparison.png')
plt.close()

print('Training complete. Artifacts saved:')
print(f" - Model: {models_dir / 'best_model.joblib'}")
print(f" - Scaler: {models_dir / 'scaler.joblib'}")
print(f" - Confusion matrix image: static/images/confusion_matrix.png")
print(f" - Accuracy chart: static/images/accuracy_comparison.png")

if __name__ == '__main__':
    pass
