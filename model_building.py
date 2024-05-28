import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.metrics import make_scorer
from sklearn import preprocessing


def preprossing(excel_name, sheet_name) -> pd.DataFrame:
    df = pd.read_excel(excel_name, sheet_name=sheet_name,dtype={"contbr_zip":str})
    df = df[["Type","POS_KS","末端15bp","人群频率_somatk",
        "local_alignment_score_mut","变异","Case_ref_readsNum","Case_var_readsNum","Ctrl_ref_readsNum","Ctrl_var_readsNum",
        "STRAND_FS","FILT_MQ_FS","FILT_BQ_FS","MQ_FS","BQ_FS","END_VAR_R","VAR_MUL_MAP_R","REF_MUL_MAP_R",
        "VAR_MIS","REF_MIS","VAR_IDL","REF_IDL","VAR_SC","REF_SC"]]

    # Change index value
    df.set_index(["变异"],inplace=True) 

    # Select only samples treated with enzyme

    df = df[df["Type"] == "酶切_Only" ]
    df = df.drop(["Type"], axis=1)
    print("酶切_Only samples:",df.shape)

    # Delete the row with nonnumeric data
    for e in df.columns[0:-1]:
        df[e]=pd.to_numeric(df[e],'coerce')
    df = df.dropna()
    print("Only numeric samples:",df.shape)

    # Add var frequency
    df["Ctrl_var_freq"] = df["Ctrl_var_readsNum"] / (df["Ctrl_var_readsNum"] + df["Ctrl_ref_readsNum"])
    df["Case_var_freq"] = df["Case_var_readsNum"] / (df["Case_var_readsNum"] + df["Case_ref_readsNum"])

    # Label the samples with the population frequency. 
    # Drop those with ambiguous labels.
    conditions = [(df["人群频率_somatk"] >= 50), (df["人群频率_somatk"] < 50)&(df["人群频率_somatk"] > 5), (df["人群频率_somatk"] <= 5)]
    labels = [True,"Drop", False]
    df["label"] = np.select(conditions, labels)
    # print(df["label"].value_counts())
    df = df.drop(["人群频率_somatk"], axis=1)
    df = df[df["label"] != "Drop" ]
    df = df[df["label"] != "0"]
    print("Labeled samples:",df.shape)
    
    # Standardize the POS_KS
    df["POS_KS"] = preprocessing.scale(df["POS_KS"])

    return df

df = preprossing('91例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点(sheet2补充其他特征).xlsx',"特征提取")

# Build a random forest to see the most indicative variables
# Tree building
x, y = df.iloc[:,:-1].values, df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
feat_labels = df.columns[0:-1]
forest = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=5)

# Model optimization
grid = {"n_estimators":range(1,51,5)}
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
gsearch = GridSearchCV(estimator=forest,param_grid=grid,
                        scoring=scoring, refit = "AUC",cv=5, return_train_score=True)
gsearch.fit(x,y)
results = gsearch.cv_results_


# Build model

forest = gsearch.best_estimator_
print(forest)
forest.fit(x_train, y_train)
y_pred = forest.predict(x_train)
y_pred_test = forest.predict(x_test)

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

# draw_roc(x_test, y_test)
joblib.dump(forest, 'RF_model.pkl')
print("Accuracy in training set:",accuracy_score(y_train,y_pred))
print("Accuracy in testing set:",accuracy_score(y_test,y_pred_test))
print("Precision in training set:",precision_score(y_train,y_pred,pos_label='True'))
print("Precision in testing set:",precision_score(y_test,y_pred_test,pos_label='True'))
print("Recall in training set:",recall_score(y_train,y_pred,pos_label='True'))
print("Recall in testing set:",recall_score(y_test,y_pred_test,pos_label='True'))








