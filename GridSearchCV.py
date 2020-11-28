# ●●●●●　グリッドサーチ編　　●●●●●●●●●●
#IRISデータ　①
from sklearn.datasets import load_iris
iris = load_iris()

#訓練データとテストデータを分割 ②
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(iris.data,iris.target,random_state=0)

#★テストするモデル、交差検証、グリッドサーチCVの定義　③★★
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#グラフ系 ④ 
import matplotlib.pyplot as plt
import mglearn.plots as mplt
import mglearn.tools as mtool

#PandasとNumpy ⑤
import pandas as pd
import numpy as np

# Forで実装するとこんな感じ
#ﾊﾟﾗﾒｰﾀの初期設定
best_score=0;best_parameters = {}
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma,C=C)
        scores = cross_val_score(svm, xtrain, ytrain,cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'C':C,'gamma':gamma}

svm = SVC(**best_parameters) 
svm.fit(xtrain, ytrain) #再構築
mplt.plot_cross_val_selection();plt.show()

#GridSearchCVを使う場合
paramGrid = {'C':[0.001, 0.01, 0.1, 1, 10, 100],'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
gridSearch = GridSearchCV(SVC(), paramGrid, cv=5)
gridSearch.fit(xtrain,ytrain)
gridSearch.best_params_
gridSearch.score(xtest,ytest)
results = pd.DataFrame(gridSearch.cv_results_)
scores = np.array(results.mean_test_score).reshape(6,6)
#検証スコアをCとgammawのヒートマップで可視化
mtool.heatmap(scores,xlabel='gamma',xticklabels=paramGrid['gamma'],
ylabel='C',yticklabels=paramGrid['C'],cmap="viridis");plt.show()

#サーチグリットのパラメータ範囲が不適切な場合のヒートマップによる可視化
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
param_grid_linear = {'C':np.linspace(1, 2, 6),'gamma':np.linspace(1, 2, 6)}
param_grid_one_log = {'C':np.linspace(1, 2, 6),'gamma':np.logspace(-3, 2, 6)}
param_grid_range = {'C':np.logspace(-3, 2, 6),'gamma':np.logspace(-7, -2, 6)}
for param_grid, ax in zip([param_grid_linear, param_grid_one_log, param_grid_range],axes):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(xtrain,ytrain)
    scores = grid_search.cv_results_['mean_test_score'].reshape(6,6)
    scores_image = mtool.heatmap(scores, xlabel='gamma',ylabel='C', xticklabels=param_grid['gamma'],
    yticklabels=param_grid['C'], cmap="viridis",ax=ax)
plt.colorbar(scores_image, ax=axes.tolist());plt.show()

#グリットでないサーチ空間でも計算可能 param_gridに注目！
param = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = [{'kernel':['rbf'],'C':param,'gamma':param},{'kernel':['linear'],'C':param}]
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(xtrain, ytrain)
print(grid_search.best_params_);print(grid_search.best_score_)
results = pd.DataFrame(grid_search.cv_results_)
results.tail()

#① 訓練データとテストデータの抽出を５回
#② 訓練データに対してGridSearchCVをかける（６X６X５） 　全部で９００回計算
scores = cross_val_score(GridSearchCV(SVC(),paramGrid, cv=5), iris.data, iris.target, cv=5)
print(scores);print(scores.mean())
