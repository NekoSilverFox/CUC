# ------*------ coding: utf-8 ------*------
# @Time    : 2023/3/27 18:45
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : CUC
# @File    : main.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------

from CodingUnitClassifier import CodingUnitClassifier
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import dataset
import time
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    # 划分数据集
    # x_train, x_test, y_train, y_test = dataset_helix()  # 小规模对角正态分布
    # x_train, x_test, y_train, y_test = test_data_a()  # 小规模对角正态分布
    # x_train, x_test, y_train, y_test = dataset.sawtooth()  # 锯齿

    x_train, x_test, y_train, y_test = dataset.fourclass()  # 蛇形
    # x_train, x_test, y_train, y_test = dataset_nesting()  # 甜甜圈

    # 归一化
    transfer = MinMaxScaler(feature_range=(0, 10))
    x_train = transfer.fit_transform(X=x_train)
    x_test = transfer.transform(X=x_test)

    arr_score = []
    arr_time = []

    cre = 2
    t = 0.90

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
