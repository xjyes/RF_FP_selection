from matplotlib import pyplot as plt
from model_building import feat_labels
import numpy as np
import joblib

forest = joblib.load("RF_model.pkl")
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]


feat_labels = feat_labels[indices]
importances = importances[indices]
color = []

for i in importances:
    if i >= 0.15:
        color.append("r")
    elif 0.10 <= i < 0.15:
        color.append("b")
    else:
        color.append("grey")

print(feat_labels)

plt.figure(figsize=(12,6))
plt.title("Importance features -- top10",
          fontsize=12)
plt.bar(range(len(importances[0:10])), importances[0:10], tick_label= feat_labels[0:10], color = color, width = 0.3, alpha = 0.5)
plt.xticks(rotation = 15)
plt.axhline(0.15, c = "r", linewidth=0.3, ls= "--", label = "cut: 0.15")
plt.axhline(0.1, c = "b", linewidth=0.3, ls= "--", label = "cut: 0.10")
plt.tick_params(axis='x', labelsize=8)

plt.legend()
plt.show()
