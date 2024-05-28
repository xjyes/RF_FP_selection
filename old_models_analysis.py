import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df1_name = "91例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点(sheet2补充其他特征).xlsx"
sheet1_name = "特征提取"

df1 = pd.read_excel(df1_name, sheet_name=sheet1_name,dtype={"contbr_zip":str})
df1 = df1[["最终保留", "Type", "变异"]]

# Change index value
df1.set_index(["变异"],inplace=True) 


# Rename column
df1 = df1[df1["Type"] != "酶切_Only_Notin688" ]

# In this column, True means false-positive (real mutation) and False means true-positive (fake mutation).
numbers1 = [["最终保留","Both_in_536",False],["最终保留","Both_in_536",True],["最终保留","酶切_Only",False],["最终保留","酶切_Only",True]]

df2_name = "47例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点_其他特征提取_过滤结果统计.xlsx"
sheet2_name = "01.未添加过滤逻辑_753酶切分类结果"

# df3_name = "47例平行测试样本15bp末端比例及KS相关-增加alignment.xlsx"
# sheet3_name = "common_vars"

# Used for old model analysis
df2 = pd.read_excel(df2_name, sheet_name=sheet2_name,dtype={"contbr_zip":str})
df2 = df2[["最终保留", "Type", "样本-变异"]]

# Used for artifact2 analysis
# df2 = pd.read_excel(df2_name, sheet_name=sheet2_name,dtype={"contbr_zip":str})
# df2 = df2[["artifcat2", "Type", "样本-变异", "Sam", "Hugo_Symbol", "原来打分"]]
# df2 = df2.rename(columns={"Hugo_Symbol": "Gene"})
# df2 = df2.sort_values(by=["Sam","Gene","原来打分"])

# df3 = pd.read_excel(df3_name, sheet_name=sheet3_name,dtype={"contbr_zip":str})
# df3 = df3[["Sam","原来打分","artifcat2","Gene"]]
# df3 = df3.sort_values(by=["Sam","Gene","原来打分"])

# common = df2[df2["Type"] == "same_in_536"]
# df2 = df2[df2["Type"] != "same_in_536"]
# common = common.reset_index(drop=True)
# df3 = df3.reset_index(drop=True)
# common["artifcat2"] = df3["artifcat2"]
# df2 = pd.concat([df2,common])
# df2 = df2.drop(["Sam","Gene","原来打分"],axis=1)
# df2 = df2.reset_index(drop=True)

# Change index value
df2.set_index(["样本-变异"],inplace=True) 


# Rename column
df2 = df2[df2["Type"] != "new_Only_in_536_Notin688" ]

# In this column, True means false-positive (real mutation) and False means true-positive (fake mutation).
numbers2 = [["最终保留","same_in_536",0.0],["最终保留","same_in_536",1.0],["最终保留","new_Only_in_536_in688",0.0],["最终保留","new_Only_in_536_in688",1.0]]

# prediction result for training sets
def return_value1(l) -> int:
    global df1
    label_pred = l[0]
    sampletype = l[1]
    label = l[2]

    try:
        out = df1[df1["Type"] == sampletype][label_pred].value_counts()[label]
        return out
    except:
        return 0

# prediction result for testing set   
def return_value2(l) -> int:
    global df2
    label_pred = l[0]
    sampletype = l[1]
    label = l[2]

    try:
        out = df2[df2["Type"] == sampletype][label_pred].value_counts()[label]
        return out
    except:
        return 0

result1 = list(map(return_value1, numbers1))
result2 = list(map(return_value2, numbers2))
type_label = ["Both", "酶切_Only"]
true_pred = [result1[0], result1[2]]
false_pred = [result1[1], result1[3]]

# bar plot
plt.figure()
plt.title("Prediction label ratio (91 samples, old model)",
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

type_label = ["Both", "酶切_Only"]
true_pred = [result2[0], result2[2]]
false_pred = [result2[1], result2[3]]

plt.figure()
plt.title("Prediction label ratio (47 samples, old model)",
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

