import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

import pandas as pd
train_data = pd.read_csv("/kaggle/input/porto-seguro-safe-driver-prediction/train.csv")
test_data =pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
#print(train_data)

X_train = train_data.drop('target',axis=1)
y_train = train_data['target']
#print(X_train)

X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
#supervise transform based on GBDT
grd = GradientBoostingClassifier(n_estimators=10)
grd.fit(X_train, y_train)
#one hot
grd_enc = OneHotEncoder(categories='auto')

temp = grd.apply(X_train)
np.set_printoptions(threshold=np.inf)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
print('grd_enc.get_feature_names is', grd_enc.get_feature_names())
#using onehot encoding as features, training LR
grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

#using LR predict
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(test_data)[:, :, 0]))[:, 1]
print(y_pred_grd_lm)

test_data['target']=y_pred_grd_lm

# 转化为二分类输出
test_data['target']=test_data['target'].map(lambda x:1 if x>=0.5 else 0)
#test_data[['target']].to_csv('submit_gbdt_lr.csv')
print(test_data['target'])
