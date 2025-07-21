from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt

# 1. Generate synthetic binary classification data
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,  # ↑ Better separation between classes = higher AUC
    flip_y=0.01,
    random_state=42
)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]  # For ROC

# 5. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# 6. Plot ROC Curve
plt.figure(figsize=(14, 6))

# Subplot 1: ROC
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'r--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – AUC > 0.8")
plt.legend()
plt.grid(True)

# Subplot 2: Confusion Matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(ax=plt.gca(), cmap='Blues')
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()


#
# 🔵 Blue Curve – The Model’s Performance
# It plots the True Positive Rate (TPR) vs False Positive Rate (FPR) at various thresholds.
#
# It shows how well the classifier distinguishes between the two classes.
#
# The more it hugs the top-left corner, the better the classifier is.
#
# The area under the blue curve (AUC) quantifies overall performance.
#
# ✅ Ideal case:
#
# Steep rise near the Y-axis, almost flat on the top — high TPR with low FPR.
#
# 🔴 Red Dashed Line – The Random Classifier Baseline
# This is the diagonal line from (0,0) to (1,1).
#
# It represents a classifier that makes random guesses.
#
# AUC = 0.5 for this line.
#
# It’s the baseline for comparison.
#
# 📉 If your blue curve is below this line, it means:
#
# Your model is worse than random guessing (AUC < 0.5).
#
# You might have flipped labels or an inverted model.