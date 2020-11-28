# ●●●●●　グリッドサーチをForで実装　　●●●●●●●●●●
#必要なデータ　①
from sklearn.datasets import load_iris
iris = load_iris()

#訓練データとテストデータを分割 ②
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(iris.data,
                                iris.target,random_state=0)

#テストするモデルを定義　③
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,ParameterGrid

#グラフ系 ④
import mglearn.plots as mplt
import matplotlib.pyplot as plt

#PandasとNumpy ⑤
import pandas as pd
import numpy as np

#ﾊﾟﾗﾒｰﾀｸﾞﾘｯﾄﾞの作成
param = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'C':param,'gamma':param}

#Forで実装
def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores =[]
    #外側グリッド　内側 x 5 （ ９００回）
    for training_samples, test_samples in outer_cv.split(X, y):
        best_params = {}
        best_score = -np.inf
        #内側グリッド ﾊﾟﾗﾒｰﾀｾｯﾄ分繰り返し  6 X 6
        for parameters in parameter_grid:
            cv_scores = []
            #　内側グリッド２ 
            for inner_train, inner_test in inner_cv.split(
                X[training_samples],y[training_samples]):
                clf = Classifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])
                score = clf.score(X[inner_test],y[inner_test])
                cv_scores.append(score)
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = parameters
        #SVCのような分類器の処理
        clf = Classifier(**best_params)
        clf.fit(X[training_samples],y[training_samples])
        outer_scores.append(clf.score(X[test_samples],y[test_samples]))
    return np.array(outer_scores)

#実際に実行
scores = nested_cv(iris.data, iris.target, StratifiedKFold(5),
                    StratifiedKFold(5),SVC, ParameterGrid(param_grid))
print(scores)