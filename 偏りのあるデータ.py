#●●●●●●●　ライブラリの取り込み　●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
#評価系
from sklearn.metrics import average_precision_score,f1_score,\
        roc_curve,precision_recall_curve,classification_report,\
        roc_auc_score,confusion_matrix,accuracy_score,confusion_matrix
#データ系
from sklearn.datasets import make_blobs,load_digits
digits = load_digits()
#ﾓﾃﾞﾙ系
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
#ｸﾞﾗﾌ、描画系
import matplotlib.pyplot as plt
import mglearn.plots as mplt
import mglearn.tools as mtool
#Basic
import numpy as np

#●●●●●　datasets with imbalanced classes 偏ったクラスのデータセット　●●●●●●●
#デタラメに予想してもスコアが高くなる例
y = digits.target == 9         #９か９以外の論理ベクトル作成　ｰ> 偏ったクラス
xtrain, xtest, ytrain, ytest = train_test_split(digits.data,y,random_state=0)
dummy_majority = DummyClassifier(strategy='most_frequent').fit(xtrain,ytrain)
pred_most_frequent = dummy_majority.predict(xtest)
print("予想されるラベル:{}".format(np.unique(pred_most_frequent)))
print("スコア:{:.2f}".format(dummy_majority.score(xtest,ytest)))

#●●●●●●●●●　３種類の分類スコア比較　fit -> predict -> scoreの順番に注目●●●●●●●●
#Dummy に予想した分類スコア
dummy = DummyClassifier().fit(xtrain, ytrain)
pred_dummy =  dummy.predict(xtest)
print(dummy.score(xtest,ytest)) #dummy Score

#Descition Trdee（深度２）で予測した分類スコア
tree = DecisionTreeClassifier(max_depth=2).fit(xtrain,ytrain)
pred_tree = tree.predict(xtest)
print(tree.score(xtest,ytest)) #Tree Test Score

#ロジスティック回帰で予想した分類スコア
logreg = LogisticRegression(C=0.1).fit(xtrain,ytrain)
pred_logreg = logreg.predict(xtest)
print(logreg.score(xtest,ytest))     #logreg score

#●●●●　混合行列（Confusion Matrixで見る）で可視化する　●●●●●●●
print(confusion_matrix(ytest,pred_most_frequent)) #dummy Most Frequent
print(confusion_matrix(ytest,pred_dummy))         #dymmy 
print(confusion_matrix(ytest,pred_tree))          #Decision Tree
print(confusion_matrix(ytest,pred_logreg))        #Logistic Regression

#Confusin Matrix (大きく表示 Ligreg)
mplt.plot_confusion_matrix_illustration();plt.show()
#TP,FP,TN,FN
mplt.plot_binary_confusion_matrix();plt.show()

#●●●● classification_report , threshold , decision_function　●●●●●
x, y = make_blobs(n_samples=(4000,500), cluster_std=[7.0,2], random_state=22)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state=0)
svc = SVC(gamma=0.05).fit(xtrain,ytrain)
classification_report(ytest,svc.predict(xtest)) #classification_report
y_pred_lower_threshold = svc.decision_function(xtest) > -.8
mplt.plot_decision_threshold();plt.show()

#precision recall カーブ 　SVC（gamma=0.05)  （適合率、再現率カーブ）
precision, recall, threshholds = precision_recall_curve(ytest, svc.decision_function(xtest))
close_zero = np.argmin(np.abs(threshholds))
plt.plot(precision[close_zero],recall[close_zero],'o',markersize=10,\
                  label="threshold zero", fillstyle="none", c='k', mew=2)
plt.plot(precision,recall, label="precision recall curve")
plt.xlabel("Precision");plt.ylabel("Rcall");plt.show()

#【SVM VS RandomForest】  precision_recall_curve でモデル比較
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(xtrain, ytrain)
#rf.predict_proba(xtest)[:,1]はサンプルがクラス１になる確率（１列め）を示します
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(ytest, rf.predict_proba(xtest)[:,1])
plt.plot(precision,recall, label="SVC")
plt.plot(precision[close_zero],recall[close_zero],'o',markersize=10,\
               label="threshold zero svc",fillstyle="none",c='k',mew=2)
plt.plot(precision_rf, recall_rf, label="rf")
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^',c='k',\
         markersize=10,label="thresholds 0.5 rf", fillstyle="none", mew=2)
plt.xlabel("Precision");plt.ylabel("Recall");plt.legend(loc="best")
plt.show()
#f1 score
print(f1_score(ytest,rf.predict(xtest)))     #f1 score of random forest
print(f1_score(ytest,svc.predict(xtest)))   #fi score of svc

#average_precision_score -> 面積　（f1 scoreよりわかりやすい）
ap_rf = average_precision_score(ytest,rf.predict_proba(xtest)[:,1])
ap_svc = average_precision_score(ytest,svc.decision_function(xtest))
print("{:.3f}".format(ap_rf)) #Average precision of random forest
print("{:.3f}".format(ap_svc)) #Average precision of SVC

#★★ roc_curve ★★ TPR、FPRカーブ （引数としてdecision_functionかpredict_proba
fpr, tpr, threshholds = roc_curve(ytest, svc.decision_function(xtest)) #ROCカーブ
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR");plt.ylabel("TPR(recall)")
close_zero = np.argmin(np.abs(threshholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,\
     label="threshold zero", fillstyle="none",c='k', mew=2)
plt.legend(loc=4);plt.show()

#【SVM VS RandomForest】  roc_curve 比較  predict_proba
fpr_rf, tpr_rf, threshholds = roc_curve(ytest,rf.predict_proba(xtest)[:,1])#ROCカーブ
plt.plot(fpr, tpr, label="ROC Curve SVC")
plt.plot(fpr_rf, tpr_rf, label="ROC Curve RF")
plt.xlabel("FPR");plt.ylabel("TPR(recall)")
plt.plot(fpr[close_zero],tpr[close_zero],'o',markersize=10,\
    label="threshold zero SVC",fillstyle="none", c='k', mew=2)
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(fpr_rf[close_default_rf],tpr[close_default_rf],'^',markersize=10,\
    label="threshold 0.5 RF", fillstyle="none",c='k',mew=2)
plt.legend(loc=4);plt.show()

#roc_auc_score関数で計算 -> 面積
rf_auc = roc_auc_score(ytest, rf.predict_proba(xtest)[:,1]) #roc_auc_score
svc_auc = roc_auc_score(ytest,svc.decision_function(xtest)) #roc_auc_score
print("{:.3f}".format(rf_auc))  #AUC for Random Forest
print("{:.3f}".format(svc_auc)) #AUC for SVC: 

#再び SVCモデルで ９以外のクラス分類問題へ 
digits = load_digits()
y = digits.target==9
xtrain, xtest, ytrain, ytest = train_test_split(digits.data,y,random_state=0)
plt.figure()
for gamma in [1, 0.05, 0.01]:
    svc =SVC(gamma=gamma).fit(xtrain, ytrain)
    accuracy =svc.score(xtest, ytest)
    auc = roc_auc_score(ytest, svc.decision_function(xtest)) #roc_auc_score
    fpr, tpr, _ = roc_curve(ytest, svc.decision_function(xtest)) #roc_curve
    print("gamma = {:.2f} accuracy ={:.2f} AUC= {:.2f}".format(gamma, accuracy, auc))
    plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))

plt.xlabel("FPR");plt.ylabel("TPR");plt.xlim(-0.01,1);plt.ylim(0,1.02)
plt.legend(loc="best");plt.show()

#●●●●●●　多クラス分類（数字データ：０から９）●●●●●●●●●●●●●●●●●
xtrain, xtest, ytrain, ytest = train_test_split(digits.data,digits.target,random_state=0)
#ﾛｼﾞｽﾃｨｯｸ回帰で分類予測
pred = LogisticRegression().fit(xtrain,ytrain).predict(xtest)
#Accuracy Score
print("Accuracy: {:.3f}".format(accuracy_score(ytest,pred)))
#混合行列
print("Confution matrix \n{}".format(confusion_matrix(ytest,pred)))

#他クラスの混同行列を描画
scores_image = mtool.heatmap(confusion_matrix(ytest,pred),xlabel='Predicted label',\
    ylabel='True label', xticklabels=digits.target_names, yticklabels=digits.target_names,\
        cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confution matrix");plt.gca().invert_yaxis();plt.show()

#それぞれにクラスに対して適合率、再現率、f値を計算
print(classification_report(ytest,pred))
#f1ｽｺｱにミクロ平均を指定
print("f1 score ﾐｸﾛ平均: {:.3f}".format(f1_score(ytest,pred,average="micro")))
#f1ｽｺｱにマクロ平均
print("f1 score ﾏｸﾛ平均: {:.3f}".format(f1_score(ytest,pred,average="macro")))

#cross_val_scoreでｽｺｱ方法　ﾃﾞﾌｫﾙﾄは【accuracy】
digits = load_digits()
print("ﾃﾞﾌｫﾙﾄのｽｺｱ :{}".format(cross_val_score(SVC(),digits.data, digits.target ==9)))
explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target==9,scoring="accuracy")
print("accuracy ｽｺｱ:{}".format(explicit_accuracy))  
#corss_val_scoreのｽｺｱ方法を【roc_auc】に変更  
roc_auc = cross_val_score(SVC(), digits.data, digits.target==9,scoring="roc_auc")
print("AUC ｽｺｱ:{}".format(roc_auc))

#gridSerchCVの結果からdecision_functionを取り出して　roc_auc_scoreを計算
xtrain, xtest, ytrain, ytest = train_test_split(digits.data,digits.target==9,random_state=0)
param_grid = {'gamma':[0.0001,0.001, 0.1, 1, 10]}
grid = GridSearchCV(SVC(), param_grid= param_grid)
grid.fit(xtrain,ytrain)
print("Best parameters:",grid.best_params_)
print("Test set AUC: {:.3f}".format(roc_auc_score(ytest,grid.decision_function(xtest))))
print("Test set accuracy: {:.3f}".format(grid.score(xtest,ytest)))

#GridSerachCVでの評価基準を【roc_auc】スコアに変更
grid = GridSearchCV(SVC(), param_grid= param_grid, scoring="roc_auc")
grid.fit(xtrain,ytrain)
print("Best parameters:",grid.best_params_)
print("Best cross-validation score (auc):{:.3f}".format(grid.best_score_))
print("Test set AUC: {:.3f}".format(grid.score(xtest,ytest)))

#利用できるスコアの種類をリスト
from sklearn.metrics.scorer import SCORERS
print("Available scores \n{}".format(sorted(SCORERS.keys())))
