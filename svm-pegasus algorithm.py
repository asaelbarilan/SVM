''' svm version :1  date: 5/02 18:00'''
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin

class Pegasus_SVM(BaseEstimator, ClassifierMixin):
    def __init__(self,iterations=100,lambda_=0.5):
        self.iterations=iterations
        self.lambda_=lambda_


    def fit(self,x,y):
        self.x=x
        self.y=y
        self.w=np.ones((1,x.shape[1]))

        for i in range(self.iterations):
            miu = (1 / (self.lambda_ * (i + 1)))
            j=np.random.randint(0,x.shape[0])
            if y[j]*np.dot(self.w,x[j])<1:
                self.w=(1-(1/(i+1)))*self.w+miu*y[j]*np.dot(self.w,x[j])
            elif y[j]*np.dot(self.w,x[j])>=1:
                self.w=(1-miu)*self.w
        return

    def predict(self,x):
        pred=np.dot(x,self.w.T)
        return np.sign(pred)
    def decision_function(self,x):
        pred=np.dot(x,self.w.T)
        return pred



if __name__ == "__main__":


    # data = load_breast_cancer
    # X, y = load_breast_cancer(return_X_y=True)
    # y = y.reshape(-1, 1)
    # y[y == 0] = -1
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    #
    # '''normalizing and scaling '''
    # ssx = StandardScaler().fit(X_train)
    # X_train_std = ssx.transform(X_train)
    # X_test_std = ssx.transform(X_test)
    # #ssy = StandardScaler().fit(y_train)
    # #y_train_std = ssy.transform(y_train)
    # #y_test_std = ssy.transform(y_test)
    # X_t = ssx.transform(X)
    # ones = np.ones(X_t.shape[0]).reshape(-1, 1)
    # X_t = np.concatenate((ones, X_t), axis=1)
    #
    # '''preproccessing-adding column for bias term '''
    # ones = np.ones(X_train_std.shape[0]).reshape(-1, 1)
    # X_train_std = np.concatenate((ones, X_train_std), axis=1)
    # ones = np.ones(X_test_std.shape[0]).reshape(-1, 1)
    # X_test_std = np.concatenate((ones, X_test_std), axis=1)
    #
    # LS = LinearSVC()
    # LS.fit(X_train_std, y_train)
    # y_pred = LS.predict(X_test_std)
    #
    # svm = Pegasus_SVM(100, 0.6)
    # svm.fit(X_train_std, y_train)
    # y_hat_test = svm.predict(X_test_std)
    #
    # print('our  prediction', np.mean(cross_val_score(svm, X_t, y, scoring='accuracy', cv=3)))
    # print('sklearn svm  prediction ', np.mean(cross_val_score(LS, X_t, y, scoring='accuracy', cv=3)))
    #
    # print('our  prediction', sklearn.metrics.accuracy_score(y_test, y_hat_test.reshape(-1, 1)))
    # print('sklearn svm  prediction ', sklearn.metrics.accuracy_score(y_test, y_pred))
    # print('bp')
    #
    # #plot the learning curve:
    # from sklearn.model_selection import validation_curve
    #
    # param_range=np.arange(0.1,1,0.1)
    # #train_sizes, train_scores, test_scores= learning_curve(Pegasus_SVM(),X,y,cv=len(train_sizes),scoring='accuracy',n_jobs=1,train_sizes=train_sizes,shuffle=True)
    # train_scores, test_scores= validation_curve(Pegasus_SVM(),X_t,y,param_name='lambda_',param_range=param_range,scoring='accuracy',n_jobs=1)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    #
    # plt.title("Validation Curve with Pegasus_SVM")
    # plt.xlabel(r"$\lambda$")
    # plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    # lw = 2
    # plt.semilogx(param_range, train_scores_mean, label="Training score",
    #             color="darkorange", lw=lw)
    # plt.fill_between(param_range, train_scores_mean - train_scores_std,
    #                 train_scores_mean + train_scores_std, alpha=0.2,
    #                 color="darkorange", lw=lw)
    # plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
    #             color="navy", lw=lw)
    # plt.fill_between(param_range, test_scores_mean - test_scores_std,
    #                 test_scores_mean + test_scores_std, alpha=0.2,
    #                 color="navy", lw=lw)
    # plt.legend(loc="best")
    # plt.show()
    # print('bp')

    from sklearn.datasets import make_blobs

    d = 2


    def make_data(d=5, n_samples=1000, imbalance=0.1):
        centers = [[-d / 2, 0],[d / 2, 0]]
        clusters_std = [(1 / 2),(1 / 2 + 1.5) ]
        X, y = make_blobs(n_samples=[int(imbalance * n_samples),int((1 - imbalance) * n_samples) ],
                          centers=centers,
                          cluster_std=clusters_std,
                          n_features=2,
                          random_state=0, shuffle=False)
        return X, y


    X, y = make_data(d=5, n_samples=1000, imbalance=0.1)

    print('bp')