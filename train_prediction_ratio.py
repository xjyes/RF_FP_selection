from model_building import preprossing, draw_roc
from feature_importance import indices
from sklearn.metrics import *
from sklearn import preprocessing
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import joblib

df_name = "91例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点(sheet2补充其他特征).xlsx"
sheet1_name = "特征提取"

df = pd.read_excel(df_name, sheet_name=sheet1_name,dtype={"contbr_zip":str})
df = df[["Type","POS_KS","末端15bp","人群频率_somatk",
        "local_alignment_score_mut","变异","Case_ref_readsNum","Case_var_readsNum","Ctrl_ref_readsNum","Ctrl_var_readsNum",
        "STRAND_FS","FILT_MQ_FS","FILT_BQ_FS","MQ_FS","BQ_FS","END_VAR_R","VAR_MUL_MAP_R","REF_MUL_MAP_R",
        "VAR_MIS","REF_MIS","VAR_IDL","REF_IDL","VAR_SC","REF_SC"]]

# Change index value
df.set_index(["变异"],inplace=True) 


# Rename column
df = df[df["Type"] != "酶切_Only_Notin688" ]
replace_dict = {"Both_in_536":1, "酶切_Only":0}
df = df.replace({"Type":replace_dict})


# Delete the row with nonnumeric data
for e in df.columns[:]:
    df[e]=pd.to_numeric(df[e],'coerce')
df = df.dropna()
print("Only numeric samples:",df.shape)
print("sample type ratio:", df["Type"].value_counts())

# Add var frequency
df["Ctrl_var_freq"] = df["Ctrl_var_readsNum"] / (df["Ctrl_var_readsNum"] + df["Ctrl_ref_readsNum"])
df["Case_var_freq"] = df["Case_var_readsNum"] / (df["Case_var_readsNum"] + df["Case_ref_readsNum"])

types = df["Type"]
df = df.drop(["Type"], axis=1)
print("酶切_Only samples:",df.shape)

# Standardize the POS_KS
df["POS_KS"] = preprocessing.scale(df["POS_KS"])

# Label the samples with the population frequency. 
# Drop those with ambiguous labels.
conditions = [(df["人群频率_somatk"] >= 50), (df["人群频率_somatk"] < 50)&(df["人群频率_somatk"] > 5), (df["人群频率_somatk"] <= 5)]
labels = [True,"Drop", False]
df["label"] = np.select(conditions, labels)
# print(df["label"].value_counts())
# population = df[df["label"] == "Drop" ]["人群频率_somatk"]
population = df["人群频率_somatk"]
df = df.drop(["人群频率_somatk"], axis=1)
grey = df[df["label"] == "Drop" ]
# df = df[df["label"] != "Drop" ]
df = df[df["label"] != "0"]
print("Labeled samples:",df.shape)


x, y = df.iloc[:,:-1].values, df.iloc[:, -1].values

forest = joblib.load("RF_model.pkl")
y_pred = forest.predict(x)
pred = pd.DataFrame(y_pred)

# print("recall:", recall_score(y,y_pred,pos_label='True'))
# print("precision:", precision_score(y,y_pred,pos_label='True'))
# print("accuracy:", accuracy_score(y,y_pred))

xg, yg = grey.iloc[:,:-1].values, grey.iloc[:, -1].values

yg_pred = forest.predict(xg)
g_pred = list(yg_pred)

# Comparison between differenct types of samples

pred = pd.DataFrame(y_pred)
pred.columns = ["pred"]
pred.index = df.index
df = pd.concat([df,pred], axis=1)
df["Type"] = types

# print(df[df["Type"] == 1]["label"].value_counts())

numbers = [["label",1,"True"],["label",1,"False"],["label",0,"True"],["label",0,"False"],
["pred",1,"True"],["pred",1,"False"],["pred",0,"True"],["pred",0,"False"]]
def return_value(l) -> int:
    global df
    label_pred = l[0]
    sampletype = l[1]
    label = l[2]

    try:
        out = df[df["Type"] == sampletype][label_pred].value_counts()[label]
        return out
    except:
        return 0

result = list(map(return_value, numbers))
print(result)
# same_label_t = df[df["Type"] == 1]["label"].value_counts()["True"]
# same_label_f = df[df["Type"] == 1]["label"].value_counts()["False"]
# new_label_t = df[df["Type"] == 0]["label"].value_counts()["True"]
# new_label_f = df[df["Type"] == 0]["label"].value_counts()["False"]

# same_pred_t = df[df["Type"] == 1]["pred"].value_counts()["True"]
# same_pred_f = df[df["Type"] == 1]["pred"].value_counts()["False"]
# new_pred_t = df[df["Type"] == 0]["pred"].value_counts()["True"]
# new_pred_f = df[df["Type"] == 0]["pred"].value_counts()["False"]

type_label = ["Both", "酶切_Only"]
true_pred = [result[4], result[6]]
false_pred = [result[5], result[7]]

plt.figure()
plt.title("Prediction label ratio (Training set)",
          fontsize=12)

plt.xlabel("Sample type")
plt.ylabel("Sample label")

plt.bar(type_label, true_pred, label='True',color='#f9766e',width=0.4)
plt.bar(type_label, false_pred,bottom=true_pred,label='False',color='#00bfc4',width=0.4)
for i in range(len(type_label)):
    plt.text(x=i, y=true_pred[i]/2, s=true_pred[i], ha='center', fontsize=10)
    plt.text(x=i, y=true_pred[i]+false_pred[i]/2, s=false_pred[i], ha='center', fontsize=10)
plt.tick_params(axis='x',length=0)
plt.grid(axis='y',alpha=0.5,ls='--')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# # grey result

# label = ["True", "False"]
# count = [g_pred.count("True"), g_pred.count("False")]
# # 画图
# fig, ax = plt.subplots()
# ax.pie(count, labels=label, autopct='%1.1f%%')

# # 添加图例和标题
# ax.legend()
# ax.set_title('Grey region prediction (Train)')

# # 显示图形
# plt.show()

# grey["population"] = population
# grey["pred"] = yg_pred
# grey["Type"] = types

# x1 = grey.loc[(grey["pred"] == "True") & (grey["Type"] == 0)]["population"]
# y1 = grey.loc[(grey["pred"] == "True") & (grey["Type"] == 0)]["local_alignment_score_mut"]

# x2 = grey.loc[(grey["pred"] == "True") & (grey["Type"] == 1)]["population"]
# y2 = grey.loc[(grey["pred"] == "True") & (grey["Type"] == 1)]["local_alignment_score_mut"]

# x3 = grey.loc[(grey["pred"] == "False") & (grey["Type"] == 0)]["population"]
# y3 = grey.loc[(grey["pred"] == "False") & (grey["Type"] == 0)]["local_alignment_score_mut"]

# x4 = grey.loc[(grey["pred"] == "False") & (grey["Type"] == 1)]["population"]
# y4 = grey.loc[(grey["pred"] == "False") & (grey["Type"] == 1)]["local_alignment_score_mut"]


# plt.figure(figsize=(8,8))
# plt.title("Grey area prediction (train) -- Both")

# plt.scatter(x1, y1, c='r', marker='o', s = 10, label='T-Both')
# plt.scatter(x2, y2, c='b', marker='o', s = 10, label='F-Both')

# plt.legend()
# plt.xlabel('population frequency')
# plt.ylabel('local alignment')
# plt.show()

# plt.figure(figsize=(8,8))
# plt.title("Grey area prediction (train) -- Only")

# plt.scatter(x3, y3, c='coral', marker='x', s = 10, label='T-Only')
# plt.scatter(x4, y4, c='c', marker='x', s = 10, label='F-Only')


# plt.legend()
# plt.xlabel('population frequency')
# plt.ylabel('local alignment')
# plt.show()


# Get the special data point ("Both" data but with "True" label)


# special = df.loc[(df["Type"] == 1) & (df["pred"] == "True")]
# label = special["label"]
# special = special.iloc[:,0:-3]
# indices = pd.DataFrame(indices)
# special = special.iloc[:,indices.loc[:,0].T]
# special["label"] = label
# special.to_excel("Both_to_True.xlsx", index=True)

# Get all data points

# replace_dict2 = {1:"Both_in_536", 0:"酶切_Only"}
# df = df.replace({"Type":replace_dict2})
# label = df["label"]
# types = df["Type"]
# pred = df["pred"]
# indices = pd.DataFrame(indices)
# df = df.iloc[:,indices.loc[:,0].T]
# df["population"] = population
# df["label"] = label
# df["Type"] = types
# df["pred"] = pred
# df.to_excel("Train_predict.xlsx", index=True)
