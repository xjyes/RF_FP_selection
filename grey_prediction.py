from model_building import preprossing, draw_roc
from sklearn.metrics import *
from sklearn import preprocessing
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import joblib

df1_name = "47例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点_其他特征提取_过滤结果统计.xlsx"
sheet1_name = "00.特征提取"

df_1 = pd.read_excel(df1_name, sheet_name=sheet1_name,dtype={"contbr_zip":str})
df_1 = df_1[["POS_KS","末端15bpreads比例","人群检出频率somatk","Type", "Sam","Gene","原来打分",
    "local_alignment_score_mut","Case_ref_readsNum","Case_var_readsNum","Ctrl_ref_readsNum","Ctrl_var_readsNum",
    "STRAND_FS","FILT_MQ_FS","FILT_BQ_FS","MQ_FS","BQ_FS","END_VAR_R","VAR_MUL_MAP_R","REF_MUL_MAP_R",
    "VAR_MIS","REF_MIS","VAR_IDL","REF_IDL","VAR_SC","REF_SC"]]
df_1 = df_1.sort_values(by=["Sam","Gene","原来打分"])

df2_name = "47例平行测试样本15bp末端比例及KS相关-增加alignment.xlsx"
sheet2_name = "common_vars"

df_2 = pd.read_excel(df2_name, sheet_name=sheet2_name, dtype={"contbr_zip":str})
df_2 = df_2[["Sam","原来打分","local_alignment_score_mut","Gene"]]
df_2 = df_2.sort_values(by=["Sam","Gene","原来打分"])

common = df_1[df_1["Type"] == "same_in_536"]
df_1 = df_1[df_1["Type"] != "same_in_536"]
common = common.reset_index(drop=True)
df_2 = df_2.reset_index(drop=True)
common["local_alignment_score_mut"] = df_2["local_alignment_score_mut"]
df = pd.concat([df_1,common])
df = df.drop(["Sam","Gene","原来打分"],axis=1)
df = df.reset_index(drop=True)

df = df[df["Type"] != "new_Only_in_536_Notin688" ]
replace_dict = {"same_in_536":1, "new_Only_in_536_in688":0}
df = df.replace({"Type":replace_dict})


# Delete the row with nonnumeric data
# for e in df.columns[0:-1]:
for e in df.columns[:]:
    df[e]=pd.to_numeric(df[e],'coerce')
df = df.dropna()
print("Only numeric samples:",df.shape)
print("sample type ratio:", df["Type"].value_counts())

# Add var frequency
df["Ctrl_var_freq"] = df["Ctrl_var_readsNum"] / (df["Ctrl_var_readsNum"] + df["Ctrl_ref_readsNum"])
df["Case_var_freq"] = df["Case_var_readsNum"] / (df["Case_var_readsNum"] + df["Case_ref_readsNum"])

# Label the samples with the population frequency. 
# Drop those with ambiguous labels.
conditions = [(df["人群检出频率somatk"] >= 50), (df["人群检出频率somatk"] < 50)&(df["人群检出频率somatk"] > 5), (df["人群检出频率somatk"] <= 5)]
labels = [True,"Drop", False]
df["label"] = np.select(conditions, labels)
# print(df["label"].value_counts())
population = df["人群检出频率somatk"]
df = df.drop(["人群检出频率somatk"], axis=1)
df = df[df["label"] == "Drop" ]
# print(df)
print("Labeled samples:",df.shape)


types = df["Type"]
df = df.drop(["Type"], axis=1)
print("酶切_Only samples:",df.shape)

# Standardize the POS_KS
df["POS_KS"] = preprocessing.scale(df["POS_KS"])

x, y = df.iloc[:,:-1].values, df.iloc[:, -1].values

forest = joblib.load("RF_model.pkl")
y_pred = forest.predict(x)
pred = list(y_pred)

# grey result

label = ["True", "False"]
count = [pred.count("True"), pred.count("False")]
# 画图
fig, ax = plt.subplots()
ax.pie(count, labels=label, autopct='%1.1f%%')

# 添加图例和标题
ax.legend()
ax.set_title('Grey region prediction (Test)')

# 显示图形
plt.show()

df["population"] = population
df["pred"] = y_pred
df["Type"] = types

x1 = df.loc[(df["pred"] == "True") & (df["Type"] == 0)]["population"]
y1 = df.loc[(df["pred"] == "True") & (df["Type"] == 0)]["local_alignment_score_mut"]

x2 = df.loc[(df["pred"] == "True") & (df["Type"] == 1)]["population"]
y2 = df.loc[(df["pred"] == "True") & (df["Type"] == 1)]["local_alignment_score_mut"]

x3 = df.loc[(df["pred"] == "False") & (df["Type"] == 0)]["population"]
y3 = df.loc[(df["pred"] == "False") & (df["Type"] == 0)]["local_alignment_score_mut"]

x4 = df.loc[(df["pred"] == "False") & (df["Type"] == 1)]["population"]
y4 = df.loc[(df["pred"] == "False") & (df["Type"] == 1)]["local_alignment_score_mut"]


plt.figure(figsize=(8,8))
plt.title("Grey area prediction -- Both")

plt.scatter(x1, y1, c='r', marker='o', s = 10, label='T-Both')
plt.scatter(x2, y2, c='b', marker='o', s = 10, label='F-Both')

plt.legend()
plt.xlabel('population frequency')
plt.ylabel('local alignment')
plt.show()


plt.figure(figsize=(8,8))
plt.title("Grey area prediction -- Only")

plt.scatter(x3, y3, c='coral', marker='x', s = 10, label='T-Only')
plt.scatter(x4, y4, c='c', marker='x', s = 10, label='F-Only')

plt.legend()
plt.xlabel('population frequency')
plt.ylabel('local alignment')
plt.show()