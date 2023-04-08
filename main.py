# ------*------ coding: utf-8 ------*------
# @Time    : 2023/3/27 18:45
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : CUC
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from CodingUnitClassifier import CodingUnitClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import pi

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
    x_1_1 = pd.Series(np.random.normal(loc=28, scale=3, size=100), name='x1')
    x_1_2 = pd.Series(np.random.normal(loc=25, scale=3, size=100), name='x2')
    y_1 = pd.Series(['blue'] * 50, name='target')

    x_2_1 = pd.Series(np.random.normal(loc=8, scale=3, size=100), name='x1')
    x_2_2 = pd.Series(np.random.normal(loc=12, scale=3, size=100), name='x2')
    y_2 = pd.Series(['red'] * 50, name='target')

    plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(x_1_1, x_1_2, s=5, c='blue')
    plt.scatter(x_2_1, x_2_2, s=5, c='red')
    plt.show()

    tmp_data_1 = pd.concat([x_1_1, x_1_2, y_1], axis=1)
    tmp_data_2 = pd.concat([x_2_1, x_2_2, y_2], axis=1)
    data = pd.concat([tmp_data_1, tmp_data_2], axis=0)
    # data = shuffle(data).reset_index(drop=True)  # 打乱样本顺序

    # 划分数据集
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1])


def test_data_a():
    data_train = pd.read_csv(filepath_or_buffer='./test_data/svmdata_a.txt', sep='\t')
    data_test = pd.read_csv(filepath_or_buffer='./test_data/svmdata_a_test.txt', sep='\t')
    data = pd.concat([data_train, data_test], axis=0)
    # data = shuffle(data).reset_index(drop=True)

    print(data)
    plt.figure(figsize=(5, 5))
    plt.scatter(x=data[data['Color'] == 'red'].iloc[:, 0], y=data[data['Color'] == 'red'].iloc[:, 1], s=5, c='red')
    plt.scatter(x=data[data['Color'] == 'green'].iloc[:, 0], y=data[data['Color'] == 'green'].iloc[:, 1], s=5,
                c='blue')
    plt.show()

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
    # data = shuffle(data).reset_index(drop=True)

    plt.figure(figsize=(5, 5))
    plt.scatter(x=data[data['Colors'] == 'red'].iloc[:, 0], y=data[data['Colors'] == 'red'].iloc[:, 1], s=5, c='red')
    plt.scatter(x=data[data['Colors'] == 'green'].iloc[:, 0], y=data[data['Colors'] == 'green'].iloc[:, 1], s=5,
                c='blue')
    plt.show()

    # return train_test_split(data.iloc[:, :-1], data.iloc[:, -1], random_state=RANDOM_STATE)
    return data_train.iloc[:, :-1], data_test.iloc[:, :-1], data_train.iloc[:, -1], data_test.iloc[:, -1]


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
    plt.savefig('./output/dataset-fourclass.png')
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

    tmp_x2 = np.random.uniform(low=3, high=7, size=200)
    tmp_y2 = np.random.uniform(low=2, high=8, size=200)
    tmp_t2 = np.full(shape=(200,), fill_value='blue')
    data_2 = pd.concat([pd.Series(tmp_x2), pd.Series(tmp_y2), pd.Series(tmp_t2)], axis=1)
    data_2.columns = ('x', 'y', 'target')
    plt.scatter(data_2['x'], data_2['y'], s=5, c='blue')
    plt.show()

    data = pd.concat([data_1, data_2], axis=0)
    # data = shuffle(data).reset_index(drop=True)
    return train_test_split(data.values[:, :-1], data.values[:, -1], random_state=RANDOM_STATE)


def dataset_checkerboard():
    num_every_board = 200  # 每个棋盘中粒子数量

    data_red_1 = pd.concat([pd.Series(np.random.uniform(low=2.5, high=5.0, size=num_every_board)),
                            pd.Series(np.random.uniform(low=0, high=2.5, size=num_every_board))], axis=1)

    data_red_2 = pd.concat([pd.Series(np.random.uniform(low=7.5, high=10, size=num_every_board)),
                            pd.Series(np.random.uniform(low=0.0, high=2.5, size=num_every_board))], axis=1)

    data_red_3 = pd.concat([pd.Series(np.random.uniform(low=0.0, high=2.5, size=num_every_board)),
                            pd.Series(np.random.uniform(low=2.5, high=5.0, size=num_every_board))], axis=1)

    data_red_4 = pd.concat([pd.Series(np.random.uniform(low=5.0, high=7.5, size=num_every_board)),
                            pd.Series(np.random.uniform(low=2.5, high=5.0, size=num_every_board))], axis=1)

    data_red_5 = pd.concat([pd.Series(np.random.uniform(low=2.5, high=5.0, size=num_every_board)),
                            pd.Series(np.random.uniform(low=5.0, high=7.5, size=num_every_board))], axis=1)

    data_red_6 = pd.concat([pd.Series(np.random.uniform(low=7.5, high=10, size=num_every_board)),
                            pd.Series(np.random.uniform(low=5.0, high=7.5, size=num_every_board))], axis=1)

    data_red_7 = pd.concat([pd.Series(np.random.uniform(low=0.0, high=2.5, size=num_every_board)),
                            pd.Series(np.random.uniform(low=7.5, high=10, size=num_every_board))], axis=1)

    data_red_8 = pd.concat([pd.Series(np.random.uniform(low=5.0, high=7.5, size=num_every_board)),
                            pd.Series(np.random.uniform(low=7.5, high=10, size=num_every_board))], axis=1)
    data_red = pd.concat([data_red_1, data_red_2, data_red_3, data_red_4,
                          data_red_5, data_red_6, data_red_7, data_red_8], axis=0)
    target_red = pd.Series(np.full(shape=(8 * num_every_board,), fill_value=0))

    data_red = data_red.reset_index(drop=True)
    data_red = pd.concat([data_red, target_red], axis=1)
    data_red.columns = ('x', 'y', 'target')

    data_blue_1 = pd.concat([pd.Series(np.random.uniform(low=0.0, high=2.5, size=num_every_board)),
                             pd.Series(np.random.uniform(low=0.0, high=2.5, size=num_every_board))], axis=1)

    data_blue_2 = pd.concat([pd.Series(np.random.uniform(low=5.0, high=7.5, size=num_every_board)),
                             pd.Series(np.random.uniform(low=0.0, high=2.5, size=num_every_board))], axis=1)

    data_blue_3 = pd.concat([pd.Series(np.random.uniform(low=2.5, high=5.0, size=num_every_board)),
                             pd.Series(np.random.uniform(low=2.5, high=5.0, size=num_every_board))], axis=1)

    data_blue_4 = pd.concat([pd.Series(np.random.uniform(low=7.5, high=10, size=num_every_board)),
                             pd.Series(np.random.uniform(low=2.5, high=5.0, size=num_every_board))], axis=1)

    data_blue_5 = pd.concat([pd.Series(np.random.uniform(low=0.0, high=2.5, size=num_every_board)),
                             pd.Series(np.random.uniform(low=5.0, high=7.5, size=num_every_board))], axis=1)

    data_blue_6 = pd.concat([pd.Series(np.random.uniform(low=5.0, high=7.5, size=num_every_board)),
                             pd.Series(np.random.uniform(low=5.0, high=7.5, size=num_every_board))], axis=1)

    data_blue_7 = pd.concat([pd.Series(np.random.uniform(low=2.5, high=5.0, size=num_every_board)),
                             pd.Series(np.random.uniform(low=7.5, high=10, size=num_every_board))], axis=1)

    data_blue_8 = pd.concat([pd.Series(np.random.uniform(low=7.5, high=10, size=num_every_board)),
                             pd.Series(np.random.uniform(low=7.5, high=10, size=num_every_board))], axis=1)
    data_blue = pd.concat([data_blue_1, data_blue_2, data_blue_3, data_blue_4,
                           data_blue_5, data_blue_6, data_blue_7, data_blue_8], axis=0)
    target_blue = pd.Series(np.full(shape=(8 * num_every_board,), fill_value=1))

    data_blue = data_blue.reset_index(drop=True)
    data_blue = pd.concat([data_blue, target_blue], axis=1)
    data_blue.columns = ('x', 'y', 'target')

    data = pd.concat([data_red, data_blue], axis=0)
    # data = shuffle(data).reset_index(drop=True)
    nn, x_test, nnn, y_test = train_test_split(data.values[:, :-1], data.values[:, -1], random_state=RANDOM_STATE)

    num_noise = 300  # 噪音数量
    noise_red = pd.concat([pd.Series(np.random.uniform(low=0.0, high=10.0, size=num_noise)),
                           pd.Series(np.random.uniform(low=0.0, high=10.0, size=num_noise)),
                           pd.Series(np.full(shape=(num_noise,), fill_value=0))], axis=1)
    noise_red.columns = ('x', 'y', 'target')
    noise_blue = pd.concat([pd.Series(np.random.uniform(low=0.0, high=10.0, size=num_noise)),
                            pd.Series(np.random.uniform(low=0.0, high=10.0, size=num_noise)),
                            pd.Series(np.full(shape=(num_noise,), fill_value=1))], axis=1)
    noise_blue.columns = ('x', 'y', 'target')
    noise = pd.concat([noise_red, noise_blue], axis=0)
    data = pd.concat([data, noise], axis=0)  # 加入噪音

    plt.figure(figsize=(5, 5))
    plt.scatter(data_red['x'], data_red['y'], s=5, c='red')
    plt.scatter(data_blue['x'], data_blue['y'], s=5, c='blue')
    plt.scatter(noise_red['x'], noise_red['y'], s=5, c='red', marker='o')
    plt.scatter(noise_blue['x'], noise_blue['y'], s=5, c='blue', marker='o')
    plt.savefig('./output/dataset-checkerboard.png')
    plt.show()

    x_train, nn, y_train, nnn = train_test_split(data.values[:, :-1], data.values[:, -1], random_state=RANDOM_STATE)

    return x_train, x_test, y_train, y_test


def dataset_helix():
    N = 400
    theta = np.sqrt(np.random.rand(N)) * 2 * pi  # np.linspace(0,2*pi,100)

    r_a = 2 * theta + pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(N, 2)

    r_b = -2 * theta - pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(N, 2)

    res_a = np.append(x_a, np.zeros((N, 1)), axis=1)  # 带有特征值和目标值
    res_b = np.append(x_b, np.ones((N, 1)), axis=1)

    data = np.append(res_a, res_b, axis=0)

    plt.figure(figsize=(5, 5))
    plt.scatter(x_a[:, 0], x_a[:, 1])
    plt.scatter(x_b[:, 0], x_b[:, 1])
    plt.show()

    return train_test_split(data[:, :-1], data[:, -1], random_state=RANDOM_STATE)

if __name__ == '__main__':
    # 划分数据集
    x_train, x_test, y_train, y_test = dataset_helix()  # 小规模对角正态分布
    # x_train, x_test, y_train, y_test = test_data_a()  # 小规模对角正态分布
    # x_train, x_test, y_train, y_test = dataset_fourclass()  # 蛇形
    # x_train, x_test, y_train, y_test = dataset_nesting()  # 甜甜圈

    # 归一化
    transfer = MinMaxScaler(feature_range=(0, 10))
    x_train = transfer.fit_transform(X=x_train)
    x_test = transfer.transform(X=x_test)

    arr_score = []
    arr_time = []

    # cre = 3
    # t = 0.84

    for cre in range(0, 4):
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>> Cre: {cre} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        for t in np.arange(start=0.70, stop=1.0, step=0.01):
            print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>> Cre: {cre},  t: {t}     开始 fit')
            blockPrint()  # 禁用 print 输出
            start_time = time.time()  # 开始时间 >>>>>>>>>>>>>>>>>
            estimator = CodingUnitClassifier(Cre=cre, threshold=t,
                                             is_draw_2D=False, color_map=('blue', 'red'), pic_save_path='./output/CUC')
            estimator.fit(X=x_train, y=y_train)
            end_time = time.time()   # <<<<<<<<<<<<<<<<< 结束时间
            enablePrint()

            arr_predict = estimator.predict(X=x_test)
            score = estimator.score(X=x_test, y=y_test)
            arr_score.append(score)
            print(f'正确率 score: {score}')

            time_use = end_time - start_time
            arr_time.append(time_use)
            print(f'Estimator 预估器耗时：{time_use}s\n')

            print(f'arr_score\n{arr_score}')
            print(f'arr_time\n{arr_time}')


            print('\n开始绘制')
            start_time = time.time()
            estimator.draw_2d(color_map=('blue', 'red'), pic_save_path=f'./output/CUC-Cre-{cre}-t-{format(t, ".2f")}',
                              title=f'Estimator CUC\n(Cre: {cre}  t: {format(t, ".2f")})')
            end_time = time.time()
            print(f'结束绘制，用时：{end_time - start_time}s')

            del estimator
