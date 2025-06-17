import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# 1. Load and inspect data
df = pd.read_csv("Dry_Bean_Dataset.csv")
print("First 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# 2. Handle missing values (if any)
if df.isnull().sum().any():
    df = df.dropna()
    print("Dropped missing values.")

# 3. Visualize class distribution
plt.figure(figsize=(8, 4))
sns.countplot(x="Class", data=df)
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Prepare features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=123
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 6. Build and train baseline pipeline
scaler = StandardScaler()
clf = LogisticRegression(max_iter=10000, random_state=123)
pipe = Pipeline([("scaler", scaler), ("clf", clf)])
pipe.fit(X_train, y_train)

# 7. Predict and evaluate baseline
y_pred = pipe.predict(X_test)
print("Sample predictions:", y_pred[:10])
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))

# 8. Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=pipe.classes_, yticklabels=pipe.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 9. ROC-AUC (macro, for multiclass)
try:
    y_test_bin = pd.get_dummies(y_test, columns=pipe.classes_)
    y_pred_prob = pipe.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test_bin, y_pred_prob, average="macro", multi_class="ovr")
    print("ROC-AUC Score (macro, OVR):", roc_auc)
except Exception as e:
    print("ROC-AUC calculation failed:", e)

# 10. Feature selection loop
f1s, accs, precs, recs = [], [], [], []
best_pipe = None
best_f1 = -1
best_k = 0
for k in range(1, X.shape[1]+1):
    fs = SelectKBest(f_classif, k=k)
    temp_pipe = Pipeline([("scaler", scaler), ("fs", fs), ("clf", clf)])
    temp_pipe.fit(X_train, y_train)
    y_pred_k = temp_pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred_k, average="macro")
    f1s.append(f1)
    accs.append(accuracy_score(y_test, y_pred_k))
    precs.append(precision_score(y_test, y_pred_k, average="macro"))
    recs.append(recall_score(y_test, y_pred_k, average="macro"))
    if f1 > best_f1:
        best_f1 = f1
        best_pipe = temp_pipe
        best_k = k

print(f"Best F1 Score: {best_f1:.4f} with k={best_k}")

# 11. Selected features for best k
mask = best_pipe.named_steps["fs"].get_support()
sel_feats = X.columns[mask]
scores = best_pipe.named_steps["fs"].scores_[mask]
print("Selected Features:", list(sel_feats))

# 12. Plot feature scores
plt.figure(figsize=(10, 6))
sns.barplot(x=sel_feats, y=scores)
plt.title(f"Top {best_k} Feature Scores")
plt.xlabel("Feature")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 13. Show sorted features
feat_df = pd.DataFrame({"Feature": sel_feats, "Score": scores})
print(feat_df.sort_values("Score", ascending=False))

# 14. Plot F1 score vs. number of features
plt.figure(figsize=(8, 5))
plt.plot(range(1, X.shape[1]+1), f1s, marker='o')
plt.title("F1 Score vs. Number of Selected Features")
plt.xlabel("Number of Features")
plt.ylabel("F1 Score (macro)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 15. Save the best pipeline
joblib.dump(best_pipe, "best_logreg_pipeline.pkl")
print("Best pipeline saved as 'best_logreg_pipeline.pkl'")
