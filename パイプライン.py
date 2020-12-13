#●●●●●●●　ライブラリの取り込み　●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
#PipeLine
from sklearn.pipeline import Pipeline,make_pipeline
#データ系　cancer、boston
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.datasets import load_boston
boston = load_boston()
#モデル系 
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#ｸﾞﾗﾌ、描画系
import matplotlib.pyplot as plt
import mglearn.plots as mplt
import mglearn.tools as mtool
#Basic
import numpy as np

#使用するデータはCancerデータで、まず訓練データ（xtrain, ytrain)、テストデータ（xtest, ytest）に分割
xtrain, xtest, ytrain, ytest = train_test_split(cancer.data,cancer.target,random_state=0)

#一般的なPipelineの作り方（タプルのリストを作っていく）
pipe_long = Pipeline([("scaler",MinMaxScaler()),("svm",SVC(C=100))])
pipe_long.steps
#名前を自動生成する場合　makepipeline（モジュールを順番に記述）
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
pipe_short.steps

#makepipelineで同じ名前がでるときの自動名前について確認してみる
pipe = make_pipeline(StandardScaler(), PCA(n_components=2),StandardScaler())
pipe.steps

pipe = Pipeline([('scaler',MinMaxScaler()),("svm",SVC())])
pipe.fit(xtrain, ytrain)
pipe.score(xtest, ytest)
#各ステップにおける値の取り出しかた　ー＞ named_steps属性
vectores = pipe.named_steps["svm"].support_vectors_
vectores.shape

#Pipeline + GridSeachCVの例
#生データとスケールデータが連動 mplt.plot_proper_processing();plt.show()
param = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'svm__C':param,'svm__gamma':param}
grid = GridSearchCV(pipe, param_grid,cv=5)
grid.fit(xtrain, ytrain) #xtrainはGridSearchCVの内部でｽｹｰﾘﾝｸﾞされる
grid.best_score_
grid.score(xtest, ytest)
grid.best_params_

#パイプラインを使わない場合 （単体の場合　結果は同じだが読みにくい）
scaler = MinMaxScaler().fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)
svm = SVC()
svm.fit(xtrain_scaled, ytrain)
xtest_scaled = scaler.transform(xtest)
svm.score(xtest_scaled,ytest)

#Pipelineを使わずにgridSeachCVを使った例
#生データとスケールデータが非連動 mplt.plot_improper_processing();plt.show()
param = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'C':param,'gamma':param}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(xtrain_scaled, ytrain)  #xtrain_scaledは訓練データの全体を含む
grid.best_score_
grid.score(xtest_scaled,ytest)
grid.best_params_

#GridSearchCV内のパイプライン属性へのアクセス
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C':[0.01, 0.1, 1, 10, 100]}
xtrain, xtest, ytrain, ytest = train_test_split(cancer.data,cancer.target,random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(xtrain, ytrain)
grid.best_estimator_
grid.best_estimator_.named_steps["logisticregression"].coef_
grid.best_estimator_.named_steps["logisticregression"].intercept_

#前処理にStandardScalerと、PolynomialFeaturesを当てはめ最後にリッジ回帰でフィットさせる
#前処理のパラメータの最適な組み合わせGridSearchCVで調べる
xtrain, xtest, ytrain, ytest = train_test_split(boston.data,boston.target,random_state=0)
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(),Ridge())
param_grid = {'polynomialfeatures__degree':[1, 2, 3],'ridge__alpha':[0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=1)
grid.fit(xtrain, ytrain)

#パラメータグリッドを適用したときはヒートマップで可視化するとよくわかる！
plt.matshow(grid.cv_results_['mean_test_score'].reshape(3,-1),vmin=0, cmap='viridis')
plt.xlabel("ridge_alpha");plt.ylabel("polynomialfeatures_degree")
plt.xticks(range(len(param_grid['ridge__alpha'])),param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])),param_grid['polynomialfeatures__degree'])
plt.show()
grid.best_params_
grid.score(xtest, ytest)

#多項式特徴量（PolynomiafFeatures)を使わないで行ったら。。（スコア変化 ｰ> 悪く）
param = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'ridge__alpha':param}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(xtrain, ytrain)
grid.score(xtest, ytest)

# Pipelineを使ったグリッドサーチによるモデル選択　（最終形）
# 分類モデル：SVC、前処理にStandardScalarで、CとGammaをパラメータグリッドで計算
# 分類モデル：RandomForestClassifierの、最大特徴量を１，２，３で、前処理なし
# の中から最適なものを選び出す処理
# Pipelineでstepe名を明示指定　
param = [0.001, 0.01, 0.1, 1, 10, 100]
pipe = Pipeline([('preprocessing', StandardScaler()),('classifier',SVC())])
param_grid = [
    {'classifier':[SVC()],'preprocessing':[StandardScaler(),None],
    'classifier__gamma':param,
    'classifier__C':param},
    {'classifier':[RandomForestClassifier(n_estimators=100)],
    'preprocessing':[None],
    'classifier__max_features':[1, 2, 3]} 
]
xtrain, xtest, ytrain, ytest = train_test_split(cancer.data,cancer.target,random_state=0)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(xtrain, ytrain)
grid.best_params_
grid.best_score_
grid.score(xtest, ytest)
