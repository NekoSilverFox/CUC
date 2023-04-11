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
    out = ax.contourf(xx, yy, Z, **params)
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
    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='sigmoid', gamma=0.7, C=C),
              svm.SVC(kernel='rbf', degree=3, C=C),
              svm.SVC(kernel='poly', degree=1, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=2, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=4, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=5, gamma='auto', C=C),)
    models = (clf.fit(X_train, y_train) for clf in models)

    # title for the plots
    titles = ('SVC with sigmoid kernel',
              'SVC with RBF(Guass) kernel',
              'SVC with Poly kernel degree=1',
              'SVC with Poly kernel degree=2',
              'SVC with Poly kernel degree=3',
              'SVC with Poly kernel degree=4',
              'SVC with Poly kernel degree=5')

    # for model in models:
    #     print(f'Score: {model.score(X=x_test, y=y_test)}')

    # Set-up 8x1 grid for plotting.
    fig, sub = plt.subplots(7, 1, figsize=(10, 40))
    plt.subplots_adjust(hspace=0.3)

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for model, title, ax in zip(models, titles, sub.flatten()):
        # 画出预测结果
        plot_contours(ax, model, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        # 把原始点画上去
        ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length', fontsize=20)
        ax.set_ylabel('Sepal width', fontsize=20)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title, fontsize=20)

    plt.show()

    pass
