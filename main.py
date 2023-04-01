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
import time
import warnings
import sys
import os

warnings.filterwarnings('ignore')

RANDOM_STATE = 90102

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


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
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1], random_state=RANDOM_STATE)


def test_data_b():
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_b.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_b_test.txt', sep='\t')
    data = pd.concat([data_train, data_test], axis=0)
    data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1], random_state=RANDOM_STATE)


def test_data_d():
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_d.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_d_test.txt', sep='\t')
    data = pd.concat([data_train, data_test], axis=0)
    data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1], random_state=RANDOM_STATE)


def test_data_e():
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_e.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_e_test.txt', sep='\t')
    data = pd.concat([data_train, data_test], axis=0)
    # data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1], random_state=RANDOM_STATE)


def dataset_fourclass():
    data = pd.read_csv(filepath_or_buffer='./test_data/fourclass.csv')
    data = data.dropna(axis=0)
    data_1 = data[data['target'] == -1.0]
    data_2 = data[data['target'] == 1.0]
    plt.figure(figsize=(5, 5))
    plt.scatter(x=data_1.iloc[:, 0], y=data_1.iloc[:, 1], s=5, c='blue')
    plt.scatter(x=data_2.iloc[:, 0], y=data_2.iloc[:, 1], s=5, c='red')
    plt.show()

    return train_test_split(data.values[:, :-1], data.values[:, -1], random_state=RANDOM_STATE)

def dataset_nesting():
    plt.figure(figsize=(5, 5))
    tmp_x1 = np.random.uniform(low=0, high=10, size=500)
    tmp_y1 = np.random.uniform(low=0, high=10, size=500)
    tmp_t1 = np.full(shape=(500,), fill_value='red')
    data_1 = pd.concat([pd.Series(tmp_x1), pd.Series(tmp_y1), pd.Series(tmp_t1)], axis=1)
    data_1.columns = ('x', 'y', 'target')
    data_1 = data_1.drop(data_1.query('x > 3 & x < 7 & y > 2 & y < 8').index, axis=0)
    plt.scatter(data_1['x'], data_1['y'], s=5, c='red')

    tmp_x2 = np.random.uniform(low=3, high=7, size=100)
    tmp_y2 = np.random.uniform(low=2, high=8, size=100)
    tmp_t2 = np.full(shape=(100,), fill_value='blue')
    data_2 = pd.concat([pd.Series(tmp_x2), pd.Series(tmp_y2), pd.Series(tmp_t2)], axis=1)
    data_2.columns = ('x', 'y', 'target')
    plt.scatter(data_2['x'], data_2['y'], s=5, c='blue')
    plt.show()

    data = pd.concat([data_1, data_2], axis=0)
    data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.values[:, :-1], data.values[:, -1], random_state=RANDOM_STATE)


if __name__ == '__main__':
    # 划分数据集
    # x_train, x_test, y_train, y_test = test_data_e()
    x_train, x_test, y_train, y_test = dataset_fourclass()


    # 归一化
    transfer = MinMaxScaler(feature_range=(0, 10))
    x_train = transfer.fit_transform(X=x_train)
    x_test = transfer.transform(X=x_test)

    # blockPrint()  # 禁用 print 输出
    start_time = time.time()  # 开始时间 >>>>>>>>>>>>>>>>>
    estimator = CodingUnitClassifier(num_refinement_splits=1, threshold_value=0.50,
                                     is_draw_2D=False, color_map=('blue', 'red'), pic_save_path='./output/CUC')
    estimator.fit(X=x_train, y=y_train)
    end_time = time.time()   # <<<<<<<<<<<<<<<<< 结束时间
    enablePrint()

    estimator.draw_2d(color_map=('blue', 'red'), pic_save_path='./output/CUC-fourclass')

    arr_predict = estimator.predict(X=x_test)
    # print(f'预测结果：\n{arr_predict}, {len(arr_predict)}')
    # print(f'正确结果：\n{y_test}, {y_test.shape[0]}')
    print(f'正确率 score:\n{estimator.score(X=x_test, y=y_test)}')
    print(f'Estimator 预估器耗时：{end_time - start_time}s')

    pass
