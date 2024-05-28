import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.metrics import make_scorer
from sklearn import preprocessing



def preprossing(excel_name, sheet_name, cols) -> pd.DataFrame:
    df = pd.read_excel(excel_name, sheet_name=sheet_name,dtype={"contbr_zip":str})
    df = df[cols]

    # Change index value
    df.set_index(["变异"],inplace=True) 

    # Select only samples treated with enzyme

    df = df[df["Type"] == "酶切_Only" ]
    df = df.drop(["Type"], axis=1)
    # print("酶切_Only samples:",df.shape)

    # Delete the row with nonnumeric data
    for e in df.columns[0:-1]:
        df[e]=pd.to_numeric(df[e],'coerce')
    df = df.dropna()
    # print("Only numeric samples:",df.shape)

    # Label the samples with the population frequency. 
    # Drop those with ambiguous labels.
    conditions = [(df["人群频率_somatk"] >= 50), (df["人群频率_somatk"] < 50)&(df["人群频率_somatk"] > 5), (df["人群频率_somatk"] <= 5)]
    labels = [True,"Drop", False]
    df["label"] = np.select(conditions, labels)
    # print(df["label"].value_counts())
    df = df.drop(["人群频率_somatk"], axis=1)
    df = df[df["label"] != "Drop" ]
    df = df[df["label"] != "0"]
    # print("Labeled samples:",df.shape)
    
    # Standardize the POS_KS
    if "POS_KS" in df.columns:
        df["POS_KS"] = preprocessing.scale(df["POS_KS"])

    return df

# Model Evaluation

def draw_roc(x_test, y_test):
    y_pred_roc = forest.predict_proba(x_test)[:, 1]
    fpr_Nb, tpr_Nb, _ = roc_curve(y_test, y_pred_roc, pos_label="True")
    aucval = auc(fpr_Nb, tpr_Nb)    
    plt.figure(figsize=(10,8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_Nb, tpr_Nb,"r",linewidth = 3)
    plt.grid()
    plt.xlabel("FP rate")
    plt.ylabel("TP rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC curve of Random Forest")
    plt.text(0.15,0.9,"AUC = "+str(round(aucval,3)))
    plt.show()

# Testing data preprocessing
df1_name = "47例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点_其他特征提取_过滤结果统计.xlsx"
sheet1_name = "00.特征提取"

df_1 = pd.read_excel(df1_name, sheet_name=sheet1_name,dtype={"contbr_zip":str})
df_1 = df_1[["POS_KS","末端15bpreads比例","人群检出频率somatk","Type", "Sam","Gene","原来打分", "样本-变异",
    "local_alignment_score_mut","VAR_SC"]]
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
df_test = pd.concat([df_1,common])
df_test = df_test.drop(["Sam","Gene","原来打分"],axis=1)
df_test = df_test.reset_index(drop=True)
df_test.set_index(["样本-变异"],inplace=True) 

# Select only samples treated with enzyme

df_test = df_test[df_test["Type"] != "new_Only_in_536_Notin688" ]
replace_dict = {"same_in_536":1, "new_Only_in_536_in688":0}
df_test = df_test.replace({"Type":replace_dict})

# Delete the row with nonnumeric data
for e in df_test.columns[0:-1]:
    df_test[e]=pd.to_numeric(df_test[e],'coerce')
df_test = df_test.dropna()

# Label the samples with the population frequency. 
# Drop those with ambiguous labels.
conditions = [(df_test["人群检出频率somatk"] >= 50), (df_test["人群检出频率somatk"] < 50)&(df_test["人群检出频率somatk"] > 5), (df_test["人群检出频率somatk"] <= 5)]
labels = [True,"Drop", False]
df_test["label"] = np.select(conditions, labels)
df_test = df_test[df_test["label"] != "0"]
df_test = df_test[df_test["label"] != "Drop" ]
df_test = df_test.drop(["人群检出频率somatk"], axis=1)
y_test = df_test.iloc[:,-1].values

# Standardize the POS_KS
if "POS_KS" in df_test.columns:
    df_test["POS_KS"] = preprocessing.scale(df_test["POS_KS"])


colnames = [["变异","Type","人群频率_somatk","POS_KS","末端15bp","local_alignment_score_mut","VAR_SC"],["变异","Type","人群频率_somatk","POS_KS","末端15bp","VAR_SC"],
            ["变异","Type","人群频率_somatk","POS_KS","local_alignment_score_mut","VAR_SC"],["变异","Type","人群频率_somatk","末端15bp","local_alignment_score_mut","VAR_SC"],
            ["变异","Type","人群频率_somatk","POS_KS","末端15bp","local_alignment_score_mut"]]
colnames2 = [["POS_KS","末端15bpreads比例","local_alignment_score_mut","VAR_SC"],["POS_KS","末端15bpreads比例","VAR_SC"],["POS_KS","local_alignment_score_mut","VAR_SC"],
             ["末端15bpreads比例","local_alignment_score_mut","VAR_SC"],["POS_KS","末端15bpreads比例","local_alignment_score_mut"]]

# Scores obtained from original model
train = [[0.9938650306748467,1.0,0.996832709473078]]
test = [[0.9965349965349966, 0.9951557093425606, 0.9941832283082889]]


for col in colnames:
    df = preprossing('91例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点(sheet2补充其他特征).xlsx',"特征提取",col)

    # Build a random forest to see the most indicative variables
    # Tree building
    x_train, y_train = df.iloc[:,:-1].values, df.iloc[:, -1].values
    feat_labels = df.columns[0:-1]
    forest = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=5)

    # Model optimization
    grid = {"n_estimators":range(1,51,5)}
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    gsearch = GridSearchCV(estimator=forest,param_grid=grid,
                            scoring=scoring, refit = "AUC",cv=5, return_train_score=True)
    gsearch.fit(x_train,y_train)
    results = gsearch.cv_results_


    # Build model
    forest = gsearch.best_estimator_
    forest.fit(x_train, y_train)

    # Prediction over train and test data
    y_pred_train = forest.predict(x_train)
    x_test = df_test[colnames2[colnames.index(col)]]
    y_pred_test = forest.predict(x_test)
    
    # Append result 
    result_train = []
    result_test = []
    result_train.append(precision_score(y_train,y_pred_train,pos_label='True'))
    result_train.append(recall_score(y_train,y_pred_train,pos_label='True'))
    result_train.append(accuracy_score(y_train,y_pred_train))
    result_test.append(precision_score(y_test,y_pred_test,pos_label='True'))
    result_test.append(recall_score(y_test,y_pred_test,pos_label='True'))
    result_test.append(accuracy_score(y_test,y_pred_test))
    train.append(result_train)
    test.append(result_test)


# Draw bar plot
labels = ["precision","recall","accuracy"]
x = np.arange(len(labels))  # x轴刻度标签位置
width = 0.1  # 柱子的宽度
# for train
plt.bar(x - 2.5*width, train[0], width, label='All')
plt.bar(x - 1.5*width, train[1], width, label='Four')
plt.bar(x - 0.5*width, train[2], width, label='No align')
plt.bar(x + 0.5*width, train[3], width, label='No 15bp')
plt.bar(x + 1.5*width, train[4], width, label='No KS')
plt.bar(x + 2.5*width, train[5], width, label='No SC')
plt.ylim((0.95,1))
plt.ylabel('Scores')
plt.title('Model comparison (train)')
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)
plt.legend(bbox_to_anchor=(1.05, 1), loc = 2) 
plt.show()

# For test
plt.bar(x - 2.5*width, test[0], width, label='All')
plt.bar(x - 1.5*width, test[1], width, label='Four')
plt.bar(x - 0.5*width, test[2], width, label='No align')
plt.bar(x + 0.5*width, test[3], width, label='No 15bp')
plt.bar(x + 1.5*width, test[4], width, label='No KS')
plt.bar(x + 2.5*width, test[5], width, label='No SC')
plt.ylim((0.8,1))
plt.ylabel('Scores')
plt.title('Model comparison (test)')
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)
plt.legend(bbox_to_anchor=(1.05, 1), loc = 2) 
plt.show()


