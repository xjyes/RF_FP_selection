import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel("result.xlsx",index_col=0)
df.reset_index()

# features = df.columns[:7]
# labels = df["pred"]
# print(features)
# print(df[df["POS_KS"] < -3])

# fig= plt.figure(figsize=(12, 8))

# # boxplot for important feature distribution
# for i, feature in enumerate(features):
#     data1 = df[df["pred"] == True][feature]
#     data2 = df[df["pred"] == False][feature]
#     ax = fig.add_subplot(2,4,i+1)
#     ax.boxplot([data1, data2],widths=0.3)
#     ax.set_title(f'{feature} ',fontsize=10)
#     ax.set_xticklabels(["True", "False"],fontsize = 8)

# fig.subplots_adjust(hspace=0.4, wspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
# fig.suptitle("Distributaion over features (train)", fontweight='bold',fontsize=12)

# plt.show()

print(df[(df["label"] == True) & (df["Case_var_freq"] <=10) & (df["Type"] == "酶切_Only")].shape[0]/df[(df["label"] == True)&(df["Type"] == "酶切_Only")].shape[0])
print(df[(df["label"] == False) & (df["Case_var_freq"] <=10) & (df["Type"] == "酶切_Only")].shape[0]/df[(df["label"] == False) &(df["Type"] == "酶切_Only")].shape[0])

af_10 = df[(df["Case_var_freq"] > 10)]
af_10.to_excel("/Users/xujingyi/Documents/BGI/酶切假阳性预测/results/random forest/AFLargerThan10_train_AFinset.xlsx",index=True)

# All data including 酶切Only and Both

# false_mut = df[df["pred"] == False]
# population_f = false_mut["Case_var_freq"]*100
# true_mut = df[df["pred"] == True]
# population_t = true_mut["Case_var_freq"]*100

# 酶切Only data
# false_mut = df[(df["pred"] == False) & (df["Type"] == "酶切_Only")]
# population_f = false_mut["population"]
# true_mut = df[(df["pred"] == True) & (df["Type"] == "酶切_Only")]
# population_t = true_mut["population"]

# fig, ax = plt.subplots()
# n, bins, patches = ax.hist(population_f, bins=20, edgecolor='black',range=(0,100))
# for i in range(len(patches)):
#     x = patches[i].get_x() + patches[i].get_width() / 2
#     y = patches[i].get_height()
#     ax.text(x, y, f'{int(n[i])}', ha='center', va='bottom',fontsize=8)
# plt.title("True mutation population frequency (train)")
# plt.xlabel("allele frequency")
# plt.ylabel("# of data")
# plt.grid(alpha=0.3)

# plt.show()

# plt.hist(population_f, bins = 20, range = (0,100), label = "true mut", color = '#00bfc4', alpha = 0.7) #预测为假，即为非假阳性，即为真变异
# plt.hist(population_t, bins = 20, range = (0,100), label = "false mut", color = '#f9766e', alpha = 0.6) # 预测为真，即为假阳性，即为假变异

# # 设置坐标轴标签和标题
# plt.title('Population frequency enzyme_only (train)')
# plt.xlabel('population frequency')
# plt.ylabel('# of data')

# plt.grid(alpha = 0.3)
# plt.legend()
# plt.show()