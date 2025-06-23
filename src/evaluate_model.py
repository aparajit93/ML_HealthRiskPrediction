import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_prep import load_and_prepare_data
from src.utils import load_model_and_threshold, save_model_and_threshold

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Evaluate XGBoost model and auto-tune threshold.")
parser.add_argument('--target_recall', type=float, default=0.9,
                    help='Target recall for threshold tuning (default: 0.9)')
args = parser.parse_args()

os.makedirs("reports", exist_ok=True)

cm_path = "reports\confusion_matrix.png"
roc_path = "reports\\roc_curve.png"
pr_path = "reports\pr_curve.png"

# Load model and threshold
model, threshold = load_model_and_threshold(
    model_path='model\\xgb_best_model.joblib',
    threshold_path='model\\threshold.txt'
)

x_train, x_test, y_train, y_test = load_and_prepare_data()

y_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_prob >= threshold).astype(int)


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Threshold = {threshold:.2f})")
plt.savefig(cm_path)
plt.show()


# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='ROC Curve (AUC)')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid()
plt.savefig(roc_path)
plt.show()

# AUC score
auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {auc_score:.4f}")

# (Optional) Precision-recall curve — tune threshold here if desired
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.plot(thresholds, recall[:-1], label='Recall', color='blue')
plt.plot(thresholds, precision[:-1], label='Precision', color='green')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision & Recall vs Threshold')
plt.legend()
plt.grid()
plt.savefig(pr_path)
plt.show()

target_recall = args.target_recall
# Identify best threshold value
idx = np.where(recall[:-1] >= target_recall)[0]
if len(idx) > 0:
    chosen_threshold = thresholds[idx[-1]]  # pick highest threshold with recall >= target
    print(f"Chosen threshold for Recall >= {target_recall}: {chosen_threshold:.2f}")
    # Save new threshold back to models/
    save_model_and_threshold(model,
                             threshold=chosen_threshold,
                             model_path='model\\xgb_best_model.joblib',
                             threshold_path='model\\threshold.txt')
    print(f"New threshold saved.")
else:
    print(f"No threshold found with that Recall level. Keeping previous threshold = {threshold:.2f}")

# Classification report
print("\n=== Classification Report ===")
class_report = classification_report(y_test, y_pred)
print(class_report)

report_md = f"""
# Model Evaluation Report


---

## Metrics

- **ROC AUC:** {auc_score:.4f}  
- **Threshold:** {chosen_threshold if 'chosen_threshold' in locals() else threshold:.4f}  

---

## Classification Report

{class_report}


---

## Confusion Matrix

![Confusion Matrix]({cm_path})

---

## ROC Curve

![ROC Curve]({roc_path})

---

## Precision-Recall Curve

![Precision-Recall Curve]({pr_path})
"""
# Write markdown to file
with open("reports\evaluation_report.md", "w") as f:
    f.write(report_md)

print("Evaluation report saved to reports/evaluation_report.md")