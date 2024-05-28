from model_building import df
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


ps_correlation = df.iloc[:,:-1].corr("pearson")

mask = np.zeros_like(ps_correlation)

print(ps_correlation)

plt.figure(figsize=(8,8))
sns.set(font_scale=0.8)
sns.heatmap(ps_correlation, mask=mask,cmap="RdBu_r", annot=True, annot_kws={'size':7}, fmt=".2f")
plt.show()


