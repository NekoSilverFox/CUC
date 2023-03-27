# ------*------ coding: utf-8 ------*------
# @Time    : 2023/3/27 18:45
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : CUC
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

from CodingUnitClassifier import *
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 获取数据集（具有正态分布的）
    x_1_1 = pd.Series(np.random.normal(loc=22, scale=4, size=50), name='x1')
    x_1_2 = pd.Series(np.random.normal(loc=24, scale=4, size=50), name='x2')
    y_1 = pd.Series(['blue'] * 50, name='target')

    x_2_1 = pd.Series(np.random.normal(loc=8, scale=3, size=50), name='x1')
    x_2_2 = pd.Series(np.random.normal(loc=12, scale=3, size=50), name='x2')
    y_2 = pd.Series(['red'] * 50, name='target')

    plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(x_1_1, x_1_2, c='blue')
    plt.scatter(x_2_1, x_2_2, c='red')
    plt.show()

    tmp_data_1 = pd.concat([x_1_1, x_1_2, y_1], axis=1)
    tmp_data_2 = pd.concat([x_2_1, x_2_2, y_2], axis=1)
    data = pd.concat([tmp_data_1, tmp_data_2], axis=0)
    data = shuffle(data).reset_index(drop=True)  # 打乱样本顺序

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1])

    # 归一化
    transfer = MinMaxScaler(feature_range=(0, 10))
    x_train = transfer.fit_transform(X=x_train)
    x_test = transfer.transform(X=x_test)

    estimator = CodingUnitClassifier(is_draw_2D=True)
    estimator.fit(X=x_train, y=y_train)
    pass
