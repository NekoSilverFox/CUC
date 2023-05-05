# ------*------ coding: utf-8 ------*------
# @Time    : 2023/4/11 15:13
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : CUC
# @File    : ClassifierTest.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import warnings


warnings.filterwarnings('ignore')


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in
    创建一个网状的点来绘制在

    Parameters
    ----------
    x: data to base x-axis meshgrid on 以X轴网格为基础的数据
    y: data to base y_train-axis meshgrid on
    h: stepsize for meshgrid, optional 网格的步长，可选

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, model, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    绘制一个分类器的决策边界

    Parameters
    ----------
    ax: matplotlib axes object
    model: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    return out

if __name__ == '__main__':
    # ------------------------------------------------------------------------------------
    # Get dataset
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_d.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_d_test.txt', sep='\t')

    # Split datasets
    X_train = data_train.iloc[:, :-1].values
    x_test = data_test.iloc[:, :-1].values
    y_train = data_train.iloc[:, -1:].replace(to_replace=['red', 'green'], value=[0, 1]).values
    y_test = data_test.iloc[:, -1:].replace(to_replace=['red', 'green'], value=[0, 1]).values

    # 归一化
    transfer = MinMaxScaler(feature_range=(0, 10))
    X_train = transfer.fit_transform(X=X_train)
    x_test = transfer.transform(X=x_test)

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors

    # param_grid = {'kernel': ('sigmoid', 'rbf', 'poly'),
    #               'degree': (2, 3, 4, 5),
    #               'gamma': (0.001, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0),
    #               'C': (0.1, 0.25, 0.5, 1.0, 5.0, 10.0, 20.0)}

    param_grid = {'kernel': ('sigmoid', 'rbf', 'poly'),
                  'degree': (2, 3, 4, 5),
                  'gamma': (0.001, 0.1, 0.25, 0.5, 1.0),
                  'C': (0.1, 0.25, 0.5, 1.0, 5.0, 10.0)}

    estimator = GridSearchCV(estimator=svm.SVC(),
                             param_grid=param_grid,
                             cv=3)

    estimator.fit(X=X_train, y=y_train)

    print(f'Best estimator: {estimator.best_estimator_}')
    print(f'Best score: {estimator.best_score_}')
    print(f'Best parameters: {estimator.best_params_}')

    # C = 1.0  # SVM regularization parameter
    # models = (svm.SVC(kernel='sigmoid', gamma=0.7, C=C),
    #           svm.SVC(kernel='rbf', degree=3, C=C),
    #           svm.SVC(kernel='poly', degree=1, gamma='auto', C=C),
    #           svm.SVC(kernel='poly', degree=2, gamma='auto', C=C),
    #           svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
    #           svm.SVC(kernel='poly', degree=4, gamma='auto', C=C),
    #           svm.SVC(kernel='poly', degree=5, gamma='auto', C=C),)
    # models = (clf.fit(X_train, y_train) for clf in models)
    #
    # # title for the plots
    # titles = ('SVC with sigmoid kernel',
    #           'SVC with RBF(Guass) kernel',
    #           'SVC with Poly kernel degree=1',
    #           'SVC with Poly kernel degree=2',
    #           'SVC with Poly kernel degree=3',
    #           'SVC with Poly kernel degree=4',
    #           'SVC with Poly kernel degree=5')
    #
    # X0, X1 = X_train[:, 0], X_train[:, 1]
    # xx, yy = make_meshgrid(X0, X1)
    #
    # for model, title in zip(models, titles):
    #     plt.figure(figsize=(5, 5))
    #     ax = plt.axes()  # 获取坐标轴对象
    #
    #     # 画出预测结果
    #     plot_contours(ax, model, xx, yy,
    #                   cmap=plt.cm.bwr, alpha=0.8)
    #     # 把原始点画上去
    #     ax.scatter(X0, X1, c=y_train, cmap=plt.cm.bwr, s=5)
    #     ax.set_xlim(xx.min(), xx.max())
    #     ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xlabel('Sepal length', fontsize=20)
    #     ax.set_ylabel('Sepal width', fontsize=20)
    #     ax.set_xticks(())
    #     ax.set_yticks(())
    #     ax.set_title(title, fontsize=20)
    #
    #     plt.show()

    pass
