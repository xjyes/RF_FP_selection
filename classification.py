import pandas as pd
import joblib
from sklearn import preprocessing


def preprossing(excel_name, sheet_name) -> pd.DataFrame:
    df0 = pd.read_excel(excel_name, sheet_name=sheet_name,dtype={"contbr_zip":str})
    df1 = df0[["POS_KS","末端15bp",
        "local_alignment_score_mut","Case_ref_readsNum","Case_var_readsNum","Ctrl_ref_readsNum","Ctrl_var_readsNum",
        "STRAND_FS","FILT_MQ_FS","FILT_BQ_FS","MQ_FS","BQ_FS","END_VAR_R","VAR_MUL_MAP_R","REF_MUL_MAP_R",
        "VAR_MIS","REF_MIS","VAR_IDL","REF_IDL","VAR_SC","REF_SC"]]


    # Delete the row with nonnumeric data
    for e in df1.columns[0:-1]:
        df1[e]=pd.to_numeric(df1[e],'coerce')
    df = df1.dropna()

    # Add var frequency
    df["Ctrl_var_freq"] = df["Ctrl_var_readsNum"] / (df["Ctrl_var_readsNum"] + df["Ctrl_ref_readsNum"])
    df["Case_var_freq"] = df["Case_var_readsNum"] / (df["Case_var_readsNum"] + df["Case_ref_readsNum"])

    # Standardize the POS_KS
    df["POS_KS"] = preprocessing.scale(df["POS_KS"])

    return df, df0

df, df0 = preprossing('91例平行测试样本15bp末端比例_KS_alignment_人群频率_靶点(sheet2补充其他特征).xlsx',"特征提取")

x= df.iloc[:,:].values
forest = joblib.load("RF_model.pkl")
y_pred = forest.predict(x)
pred = pd.DataFrame(y_pred)
pred.columns = ["label"]
pred.index = df.index
df = pd.concat([df0,pred], axis=1)
df.to_excel("result.xlsx", index=False,header=True)

