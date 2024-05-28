from model_building import preprossing, draw_roc
from feature_importance import indices
from sklearn.metrics import *
from sklearn import preprocessing
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import joblib

df1_name = "47例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点_其他特征提取_过滤结果统计.xlsx"
sheet1_name = "00.特征提取"

df_1 = pd.read_excel(df1_name, sheet_name=sheet1_name,dtype={"contbr_zip":str})
df_1 = df_1[["POS_KS","末端15bpreads比例","人群检出频率somatk","Type", "Sam","Gene","原来打分", "样本-变异",
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
df.set_index(["样本-变异"],inplace=True) 

# Select only samples treated with enzyme

df = df[df["Type"] != "new_Only_in_536_Notin688" ]
replace_dict = {"same_in_536":1, "new_Only_in_536_in688":0}
df = df.replace({"Type":replace_dict})

# print(df)



# Delete the row with nonnumeric data
for e in df.columns[0:-1]:
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
print(df["label"].value_counts())
df = df[df["label"] != "0"]
population = df["人群检出频率somatk"]
df = df.drop(["人群检出频率somatk"], axis=1)
df = df[df["label"] != "Drop" ]
print("Labeled samples:",df.shape)

types = df["Type"]
df = df.drop(["Type"], axis=1)
print("酶切_Only samples:",df.shape)

# Standardize the POS_KS
df["POS_KS"] = preprocessing.scale(df["POS_KS"])



x, y = df.iloc[:,:-1].values, df.iloc[:, -1].values

forest = joblib.load("RF_model.pkl")
y_pred = forest.predict(x)
pred = pd.DataFrame(y_pred)


# Model Evaluation

# draw_roc(x,y)
# print("Accuracy:",accuracy_score(y,y_pred))
# print("Precision",precision_score(y,y_pred,pos_label='True'))
# print("Recall:",recall_score(y,y_pred,pos_label='True'))


# Concate the original dataframe with the prediction result and sample type information

pred = pd.DataFrame(y_pred)
pred.columns = ["pred"]
pred.index = df.index
types.index = df.index
df = pd.concat([df,types],axis=1)
df = pd.concat([df,pred], axis=1)
# # print(df[df["Type"] == 1]["label"].value_counts())

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
# print(result)

# bar plot for prediction ratio

# type_label = ["Both", "酶切_Only"]
# true_pred = [result[4], result[6]]
# false_pred = [result[5], result[7]]

# plt.figure()
# plt.title("Prediction label ratio",
#           fontsize=12)

# plt.xlabel("Sample type")
# plt.ylabel("Sample label")

# plt.bar(type_label, true_pred, label='True',color='#f9766e',width=0.4)
# plt.bar(type_label, false_pred,bottom=true_pred,label='False',color='#00bfc4',width=0.4)
# for i in range(len(type_label)):
#     plt.text(x=i, y=true_pred[i]/2, s=true_pred[i], ha='center', fontsize=10)
#     plt.text(x=i, y=true_pred[i]+false_pred[i]/2, s=false_pred[i], ha='center', fontsize=10)
# plt.tick_params(axis='x',length=0)
# plt.grid(axis='y',alpha=0.5,ls='--')
# plt.legend(frameon=False)
# plt.tight_layout()
# plt.show()


# Get the special data point ("Both" data but with "True" label)

# special = df.loc[(df["Type"] == 1) & (df["pred"] == "True")]
# label = special["label"]
# special = special.iloc[:,0:-3]
# indices = pd.DataFrame(indices)
# special = special.iloc[:,indices.loc[:,0].T]
# special["label"] = label

# special.to_excel("Both_to_True_test.xlsx", index=True)

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
# df.to_excel("Test_predict.xlsx", index=True)
