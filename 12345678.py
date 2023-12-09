#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np

def normalize(series):
    return (series - series.mean(axis=0)) / series.std(axis=0)

def filterColumns(df, max_nan_proportion=0.5):
    df_train_feats = df.apply(normalize).fillna(0)
    return df_train_feats

def correlation(ser1,ser2):
    boolnotNaN=np.isfinite(ser1)&np.isfinite(ser2)
    return np.corrcoef(ser1[boolnotNaN],ser2[boolnotNaN])[0,1]


# In[1]:


import math
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from pandas.tseries.offsets import Day, MonthEnd, MonthBegin,DateOffset
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor


# In[28]:


df=pd.read_csv(r"/Users/yangjiayu/Desktop/大三上/金融计量分析/RFS/data/GHZ_ZHY_V8.csv", low_memory= False)
df.RET = pd.to_numeric(df.RET, errors='coerce')

# 删除 'RET' 列为 NaN 的行，并重置索引
df = df[~np.isnan(df.RET)].reset_index(drop=True)

# 将 'RET' 列重命名为 'next_ret'
df = df.rename(columns={'RET': 'next_ret'})



# 将所有列名转换为小写
df.columns = [name.lower() for name in df.columns]

df['year'] = [int(str(date)[:4]) for date in df.date]


# In[48]:


from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


#R_ooc square
def square_R_ooc(Y_test,Y_pred):
    #Y_pred=regressor.predict(X_test)#我想了想，还是把predict放到外面吧，在外面predict完再传进来
    R = 1 - np.sum((Y_test - Y_pred)**2) / np.sum(Y_test**2)
    return R

import matplotlib.pyplot as plt
import operator
#features importance
def features_importance(X_test,Y_test,regressor):
    Y_pred=regressor.predict(X_test)
    r0=square_R_ooc(Y_test,Y_pred)
    features=X_test.columns
    importance_dict={}
    for i in features:
        features_test=[j for j in features if j!=i]
        X_feature_test=X_test[features_test]
        Y_pred2 = regressor.predict(X_feature_test)
        r1=square_R_ooc(Y_test,Y_pred2)
        delta_r=r0-r1
        importance_dict[i]=delta_r
    top_20features=sorted(importance_dict.items(), key=operator.itemgetter(0))
    plt.bar(importance_dict)
    plt.title("Features importance")
    plt.xlabel("Reduction on R")
    plt.ylabel("Feature")
    plt.show()




# GBRT
def build_model_GBRT(X_train,y_train, lr, num, depth):
    model_GBRT = lgb.LGBMRegressor(force_row_wise='true', objective='huber',learning_rate=lr, n_estimators=num, max_depth=depth)
    model_GBRT.fit(X_train,y_train)
    return model_GBRT



feats_not_use = ["permno", "date", "year", "statpers", "sic2", "next_ret", "yr", "time_1", "time_2", "fyear","datadate","month","rdq","mve0"]
feats_to_use = [feat for feat in df.columns if feat not in feats_not_use]
y_pred = []
y_true = []
begin_year=1962
end_year=2016
first_train_gap=18
val_gap=12
test_gap=1
model=0
x=0
y=0

for i in range(end_year-begin_year+1):
    print(i)
    ind_train = df[df.year.isin(range(begin_year, begin_year+first_train_gap+i))].index
    ind_val = df[df.year.isin(range(begin_year+first_train_gap+i, begin_year+first_train_gap+i+val_gap))].index
    ind_test = df[df.year.isin(range(begin_year+first_train_gap+i+val_gap, begin_year+first_train_gap+i+val_gap+test_gap))].index

    df_train = df.loc[ind_train, :].copy().reset_index(drop=True)
    df_val = df.loc[ind_val, :].copy().reset_index(drop=True)
    df_test = df.loc[ind_test, :].copy().reset_index(drop=True)
    X_train=filterColumns(df_train[feats_to_use])
    X_val=filterColumns(df_val[feats_to_use])
    X_test=filterColumns(df_test[feats_to_use])
    Y_train=df_train["next_ret"]
    Y_val=df_val["next_ret"]
    Y_test=df_test["next_ret"]
    '''在循环里跑模型、和调超参数'''
    model_GBRT_best = build_model_GBRT(X_train, Y_train, 0.1, 20, 1)
    lr_best, num_best, depth_best, score_best = 0.1, 20, 1, model_GBRT_best.score(X_val, Y_val)
    # 网格搜索
    for j in range(1, 11):
        lr = 0.1 * j
        for num in range(20, 100, 10):
            for depth in range(1, 3):
                model_GBRT_temp = build_model_GBRT(X_train, Y_train, lr, num, depth)
                if model_GBRT_temp.score(X_val, Y_val) > score_best:
                    model_GBRT_best = model_GBRT_temp
                    lr_best, num_best, depth_best, score_best = lr, num, depth, model_GBRT_temp.score(X_val, Y_val)

    for each in model_GBRT_best.predict(X_test).flatten():
        y_pred.append(each)

    for each in Y_test:
        y_true.append(each)

    '''大循环结束'''
y_pred = np.array(y_pred)
y_true = np.array(y_true)
R2_score = square_R_ooc(y_true,y_pred)
print("R2 of model:",R2_score)
importance = features_importance(X_test, Y_test, model_GBRT_best)
