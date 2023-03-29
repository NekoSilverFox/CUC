# ------*------ coding: utf-8 ------*------
# @Time    : 2023/3/27 18:45
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : CUC
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import pandas as pd

from CodingUnitClassifier import *
import warnings

warnings.filterwarnings('ignore')


def test_data_1():
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
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1])

def test_data_a():
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_a.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_a_test.txt', sep='\t')
    data = pd.concat([data_train, data_test], axis=0)
    data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1])

def test_data_b():
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_b.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_b_test.txt', sep='\t')
    data = pd.concat([data_train, data_test], axis=0)
    data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1])

def test_data_d():
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_d.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_d_test.txt', sep='\t')
    data = pd.concat([data_train, data_test], axis=0)
    data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1])

def test_data_e():
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_e.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_e_test.txt', sep='\t')
    data = pd.concat([data_train, data_test], axis=0)
    data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1])

if __name__ == '__main__':
    # 划分数据集
    # x_train, x_test, y_train, y_test = test_data_1()
    x_train, x_test, y_train, y_test = test_data_1()


    # 归一化
    transfer = MinMaxScaler(feature_range=(0, 10))
    x_train = transfer.fit_transform(X=x_train)
    x_test = transfer.transform(X=x_test)

    estimator = CodingUnitClassifier(num_refinement_splits=1, is_draw_2D=True, color_map=('blue', 'red'), pic_save_path='./output/CUC')
    estimator.fit(X=x_train, y=y_train)

    pass
