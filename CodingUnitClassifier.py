# ------*------ coding: utf-8 ------*------
# @Time    : 2023/3/27 18:45
# @Author  : 冰糖雪狸 (NekoSilverfox)
# @Project : CUC
# @File    : CodingUnitClassifier.py
# @Software: PyCharm
# @Github  ：https://github.com/NekoSilverFox
# -----------------------------------------
import random
import copy
import numpy as np
import pandas as pd
from enum import Enum
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


class CodingUnitClassifier(object):
    """编码单元分类器预估器（estimator）"""
    class CUType(Enum):
        NOT_FINAL_CU = -1
        EMPTY_CU = -2

    def __init__(self, Cre=0, threshold=1.0, is_draw_2D=False, color_map=[], pic_save_path=None, **kw) -> None:
        """初始化

        Args:
            Cre (int, optional): 细化分割次数. Defaults to 0.
            threshold (int, optional): 临界值. Defaults to  1.0.
            is_draw_2D (bool, optional): 当绘制 2D 数据集，是否绘制中途图像. Defaults to False.
        """
        self.is_draw_2D = is_draw_2D
        self.pic_save_path = pic_save_path
        self.color_map = color_map
        self.draw_count = 0  # 绘制次数记录（还用于拓展保存时的文件名）

        self.split_count = 0  # 分割次数计数器
        self.num_refinement_splits = Cre

        if threshold > 1.0:
            self.threshold_value = 1.0  # 临界值：当某个 CU 中某种粒子占比超过这个阈值，则暂停分割
        else:
            self.threshold_value = threshold

        self.transfer_LabelEncoder = None  # 目标值的转换器（转为数字）

        self.D_train = None  # 训练集的维度数量: int
        self.X_TRAIN = None  # 不会做任何改变的特征值（训练集）
        self.Y_TRAIN = None  # 不会做任何改变的目标值（训练集）

        self.X_train = None  # 特征值（训练集）
        self.y_train = None  # 目标值（训练集）
        self.df_train = None  # 以 pandas.DataFream 形式的训练接数据（特征值和训练集是合并在一起的）。目标值的索引为 `target`，特征值的索引为 `x`，x 为从 0 开始的数字

        self.CU_min = None  # 编码单元范围最小值
        self.CU_max = None  # 编码单元范围最小值

        self.arrCU_start_points = None  # 编码单元起始点列表
        self.arrCU_dL = None  # arrCU_start_points 对应位置的编码单元的边长 `dL`
        self.arrCU_is_enable = None  # arrCU_start_points 对应位置的编码单元是否启用，True 代表启用，False 代表不启用
        self.arrCU_final_target = None  # arrCU_start_points 对应位置的编码单元的最终预测类别（-1 代表无类别）
        self.arrCU_num_point = None  # arrCU_start_points 对应位置的编码单元中粒子数量
        self.arrCU_force_infection = None  # arrCU_start_points 对应位置的编码单元的感染力度 (force of infection)

    def arr_checker(self):
        """
        检测当前 self 内数组的长度是否正常
        :return:
        """
        if self.arrCU_start_points.shape[0] != self.arrCU_dL.shape[0] != self.arrCU_is_enable.shape[0] != \
                self.arrCU_final_target.shape[0]:
            raise ValueError(f'arrCU 长度异常：\n'
                             f'\t{self.arrCU_start_points.shape[0]}'
                             f'\t{self.arrCU_dL.shape[0]}'
                             f'\t{self.arrCU_is_enable.shape[0]}'
                             f'\t{self.arrCU_final_target.shape[0]}')

    def is_point_in_CU(self, point: np.ndarray, start_point: np.ndarray, dL: float) -> bool:
        """
        判断点（样本）point，是否在编码单元中
        :param point: 需要判断的点
        :param start_point: 编码单元的起始点
        :param dL: 编码单元的边长
        :return: True-在这个 CU 中；False：不在这个 CU 中
        """
        point = np.array(point)
        start_point = np.array(start_point)
        end_point = np.array(start_point + dL)  # 结束点
        if (point > start_point).all() and (point < end_point).all():
            return True
        else:
            return False

    def predict_is_point_in_CU(self, point: np.ndarray, start_point: np.ndarray, dL: float) -> bool:
        """
        【仅供 fit 结束后预测阶段使用。因为需要预测的点可能落到编码单元的边缘】预测判断点（样本）point，是否在编码单元中
        :param point: 需要判断的点
        :param start_point: 编码单元的起始点
        :param dL: 编码单元的边长
        :return: True-在这个 CU 中；False：不在这个 CU 中
        """
        small_value = self.arrCU_dL.min() / 100  # 防止精度损失的值，算是一个容错值
        point = np.array(point)
        start_point = np.array(start_point) - small_value  # 加上值是为了可能因为精度损失导致的不正确
        end_point = np.array(start_point + dL) + small_value  # 结束点
        if (point >= start_point).all() and (point <= end_point).all():
            return True
        else:
            return False

    def is_CU_need_split(self, arr1d_start_points: np.ndarray, dL: np.float) -> np.int:
        """判断当前 CU 是否需要继续预分割（如果需要返回 -1，如果不需要返回当前 CU 所属的目标值，并且 -2 代表为空白 CU）

        Args:
            arr1d_start_points (np.ndarray): 当前 CU 的起始点
            dL (np.float): 当前 CU 的边长

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: True为需要进一步分割，如果不需要返回当前 CU 所属的目标值
        """
        if dL <= 0:
            raise ValueError(f'[CUC-ERROR] dl can not <= 0, your dL is {dL}')
        if arr1d_start_points.shape[0] != self.D_train:
            raise ValueError(
                f'[CUC-ERROR] arr1d_start_points.shape not correct: arr1d_start_points.shape is {arr1d_start_points.shape}, '
                f'should be ({self.D_train}, )')

        s_type_count = pd.Series(index=np.unique(self.y_train)).fillna(value=0)  # 计数器，列索引为类别，对应位置数据为该 CU 中此种类粒子数量
        arr_index_point_and_target = []  # 记录在当前编码单元的样本点的下标和类别

        # 遍历判断点是否在这个 CU  中
        for i in range(self.X_train.shape[0]):
            # 判断粒子在此维度上是否介于 CU 的起始点和结束点
            if self.is_point_in_CU(point=self.X_train[i], start_point=arr1d_start_points, dL=dL):
                arr_index_point_and_target.append([i, self.y_train[i]])
                s_type_count[self.y_train[i]] += 1

        # -1 代表这个编码单元里没有任何粒子，为空白编码单元
        if 0 == s_type_count.sum():
            return self.CUType.EMPTY_CU.value

        # 如果不是空 CU，则看看某种 target 的粒子占比是否达到阈值
        s_type_count = s_type_count / s_type_count.sum()  # 转换为概率
        if (s_type_count.max() >= self.threshold_value) and \
                (1 == s_type_count[s_type_count.values == s_type_count.max()].shape[0]):  # 不允许出现两种相同概率
            res_target = s_type_count[s_type_count.values == s_type_count.max()].index[0]  # 该 CU 最终的目标值

            # 如果此单元中没有异常点
            if 1.0 == s_type_count.max():
                return res_target
            # 从原数据集中移除异常点（与该 CU 目标值不一样的点）
            arr_index_point_and_target.reverse()  # 反转一下，让他从后往前删，不然删除过程中的下标变化会造成错误
            for i in arr_index_point_and_target:
                if i[1] != res_target:
                    # print(f'移除 {i}')
                    self.X_train = np.delete(self.X_train, i[0], axis=0)
                    self.y_train = np.delete(self.y_train, i[0], axis=0)
            return res_target
        else:
            return self.CUType.NOT_FINAL_CU.value

    def split_CU_and_update2arrCU(self, index_start_points: int) -> None:
        """分割当前 CU，并将分割的结果更新到方法内部成员

        Args:
            index_start_points (int): 需要分割 CU 的索引
        """
        if not self.arrCU_is_enable[index_start_points]:
            raise ValueError(f'[CUC-ERROR] CU on index {index_start_points} already disable, can not continue split')

        self.arrCU_is_enable[index_start_points] = False  # 既然对这个单元分割就说明这个单元不再使用了，因为它被差分成了许多新的小单元

        start_points = self.arrCU_start_points[index_start_points]
        new_dL = self.arrCU_dL[index_start_points] / 2
        end_points = np.array(start_points + new_dL)

        # 生成包含所有可能性组合的列表（二进制位表）
        combinations = []
        for i in range(2 ** self.D_train):
            new_combination = []
            for j in range(self.D_train):
                if i & (1 << j):
                    new_combination.append(end_points[j])
                else:
                    new_combination.append(start_points[j])
            combinations.append(new_combination)

        # 分割后的 CU 添加至缓冲区
        self.arrCU_start_points = np.vstack([self.arrCU_start_points, np.array(combinations)])
        # self.arrCU_start_points = np.array(self.arrCU_start_points.tolist() + combinations)
        self.arrCU_dL = np.hstack([self.arrCU_dL, np.full(shape=(2 ** self.D_train,), fill_value=new_dL)])
        self.arrCU_is_enable = np.hstack([self.arrCU_is_enable, np.full(shape=(2 ** self.D_train,), fill_value=True)])
        self.arrCU_final_target = np.hstack(
            [self.arrCU_final_target,
             np.full(shape=(2 ** self.D_train,), fill_value=self.arrCU_final_target[index_start_points], dtype=np.int)])

        self.split_count += 1

    def remove_disable_points(self):
        """
        删除 arr 群组中 disable 的 point
        :return:
        """
        # 移除 self arr 数组群中 disable 的 CU，
        # 同时重新标记细化分割中产生的新空白 CU（原含有粒子的大 CU 被重新切分后可能会产生不包含粒子的新 CU）
        # 同时计算 CU 的密度
        df_X_target = pd.DataFrame(self.X_train)

        new_arrCU_start_points = []  # 编码单元起始点列表
        new_arrCU_dL = []  # arrCU_start_points 对应位置的编码单元的边长 `dL`
        new_arrCU_is_enable = []  # arrCU_start_points 对应位置的编码单元是否启用，True 代表启用，False 代表不启用
        new_arrCU_final_target = []  # arrCU_start_points 对应位置的编码单元的最终预测类别（-1 代表无类别）
        new_arrCU_num_point = []  # arrCU_start_points 对应位置的编码单元中粒子数量
        new_arrCU_force_infection = []  # arrCU_start_points 对应位置的编码单元的感染力度 (force of infection)

        for i in range(self.arrCU_start_points.shape[0]):
            if not self.arrCU_is_enable[i]:
                continue
            dL = self.arrCU_dL[i]
            start_point = np.array(self.arrCU_start_points[i])
            end_point = np.array(start_point + dL)

            # 计算该单元中粒子的数量
            tmp_s_is_X_in_CU = ((df_X_target > start_point) & (df_X_target < end_point)).all(axis=1)  # all代表按行判断是否这一行都为 True（也就是 point 每个对应维度都符合），返回对应位置为 True/False 的 pd.Series
            num_point_in_CU = tmp_s_is_X_in_CU[tmp_s_is_X_in_CU == True].count()  # 该单元中粒子的数量
            target_CU = None  # 编码单元的类别
            density = None  # 密度（感染力度）

            # CU 中无粒子
            if 0 == num_point_in_CU:
                density = 0.0
                target_CU = self.CUType.EMPTY_CU.value
            else:
                density = num_point_in_CU / (dL ** self.D_train)
                target_CU = self.arrCU_final_target[i]

            new_arrCU_start_points.append(self.arrCU_start_points[i])
            new_arrCU_dL.append(self.arrCU_dL[i])
            new_arrCU_is_enable.append(self.arrCU_is_enable[i])
            new_arrCU_final_target.append(target_CU)
            new_arrCU_num_point.append(num_point_in_CU)
            new_arrCU_force_infection.append(density)

        # 释放内存
        del self.arrCU_start_points
        del self.arrCU_dL
        del self.arrCU_is_enable
        del self.arrCU_final_target
        del self.arrCU_num_point
        del self.arrCU_force_infection

        # 整理好的数组替换原本的
        self.arrCU_start_points = np.array(new_arrCU_start_points)
        self.arrCU_dL = np.array(new_arrCU_dL)
        self.arrCU_is_enable = np.array(new_arrCU_is_enable)
        self.arrCU_final_target = np.array(new_arrCU_final_target)
        self.arrCU_num_point = np.array(new_arrCU_num_point)
        self.arrCU_force_infection = np.array(new_arrCU_force_infection)

    def pre_split(self):
        """
        预分割阶段
        :return:
        """
        # 预分割阶段
        current_index = 0
        while current_index < self.arrCU_start_points.shape[0]:
            self.arr_checker()

            # 如果当前 CU 已经废弃（disable），那么就没有再对他进行处理的意义了
            if not self.arrCU_is_enable[current_index]:
                current_index += 1
                continue

            # 如果当前 CU 启用，但是已经有了最终类别
            if self.arrCU_is_enable[current_index] and self.arrCU_final_target[current_index] != self.CUType.NOT_FINAL_CU.value:
                current_index += 1
                continue

            cur_target = self.is_CU_need_split(arr1d_start_points=self.arrCU_start_points[current_index],
                                               dL=self.arrCU_dL[current_index])
            print(f'[INFO] CUC.fit(): current_index: {current_index} \t cur_target: {cur_target}')

            if cur_target == self.CUType.NOT_FINAL_CU.value:
                self.split_CU_and_update2arrCU(index_start_points=current_index)
            else:
                self.arrCU_is_enable[current_index] = True
                self.arrCU_final_target[current_index] = cur_target
                if self.D_train == 2 and self.is_draw_2D:
                    self.draw_2d(color_map=self.color_map, pic_save_path=self.pic_save_path)

            current_index += 1
            # print(f'\n>>>>>>>>>>>>>>>> 预分割阶段 {current_index}')
            # res = pd.concat([pd.DataFrame(self.arrCU_start_points),
            #                  pd.Series(self.arrCU_is_enable),
            #                  pd.Series(self.arrCU_dL),
            #                  pd.Series(self.arrCU_final_target)], axis=1)
            # res.columns = ('x', 'y', 'is_enable', 'dL', 'target')
            # print(res)

    def refinement_split(self, num: int) -> None:
        """
        细化分割
        :param num:细化分割的次数
        :return: None
        """
        if 0 == self.num_refinement_splits:
            return

        self.arr_checker()  # 执行数组长度一致性检查

        for run_time in range(num):
            # 对每个 enable 的 CU 进行细化分割
            for i in range(self.arrCU_start_points.shape[0]):
                if not self.arrCU_is_enable[i]:
                    continue
                self.split_CU_and_update2arrCU(index_start_points=i)

    def is_overlapping(self, point_1_start: float, point_1_end: float, point_2_start: float, point_2_end: float) -> bool:
        """判断某个维度的投影下，两个水平的线段是否相交

        Args:
            point_1_start (float): point1的起始点（CU某个维度上的最小值）
            point_1_end (float): point1的结束点（CU某个维度上的最大值）
            point_2_start (float): point2的起始点（CU某个维度上的最小值）
            point_2_end (float): point2的结束点（CU某个维度上的最大值）

        Returns:
            bool: True 代表相交，False 代表不相交
        """
        return (point_1_end >= point_2_start) and (point_2_end >= point_1_start)

    def infection(self) -> None:
        """
        感染阶段
        :return: 无
        """
        self.arr_checker()

        arrCU_is_been_I = np.full(shape=self.arrCU_force_infection.shape, fill_value=False)  # False 代表从来没感染过他人
        run_time = 0
        # 如果还有空白的编码单元就一直循环
        while True:
            run_time += 1

            # tmp_fi_max = -1.0
            # for i in range(self.arrCU_is_enable.shape[0]):
            #     if self.arrCU_is_been_I[i] or self.arrCU_force_infection[i] < tmp_fi_max:
            #         continue
            #     index_I = i
            #     tmp_fi_max = self.arrCU_force_infection[i]

            # 感染力度最大的单元去感染其他粒子
            # I - 感染者，他去感染其他单元
            # S - 被感染者，他被 I 所感染
            pd_beenI_and_fi = pd.DataFrame(data=[arrCU_is_been_I, self.arrCU_force_infection]).T
            sr_fi = pd_beenI_and_fi[pd_beenI_and_fi.iloc[:, 0] == False].iloc[:, 1]
            if sr_fi.shape[0] == 0:  # 都当过一次感染者
                if np.any(self.arrCU_final_target == self.CUType.EMPTY_CU.value):
                    print('[WARRING] 有的点没被感染！！！！！')
                break
            index_I = sr_fi.astype('float64').idxmax()  # 当前的感染力度最大的，且以前没做过 I 的下标

            arrCU_is_been_I[index_I] = True
            target_I = self.arrCU_final_target[index_I]
            start_point_I = self.arrCU_start_points[index_I]
            dL = self.arrCU_dL[index_I]
            end_point_I = np.array(start_point_I + dL)

            # 遍历所有 CU，当前感染者I 要去感染CU（被感染者 S）的 index
            arr_index_S = [index_I]  # 被感染者下标
            sum_point = self.arrCU_num_point[index_I]  # 所有被感染者CU中的粒子数量
            sum_V = (self.arrCU_dL[index_I] ** self.D_train)  # 所有被感染者CU中的体积
            for i in range(self.arrCU_start_points.shape[0]):
                # 当前 CU 已经是别的target，并且不是空白 CU
                if (self.arrCU_final_target[i] != target_I) and (self.arrCU_final_target[i] != self.CUType.EMPTY_CU.value):
                    continue
                # 不能自己感染自己
                if i == index_I:
                    continue

                start_point_S = self.arrCU_start_points[i]
                end_point_S = np.array(start_point_S + self.arrCU_dL[i])

                # 判断每一维度（数列-D 表示特征）的投影是否都相交
                same_d_count = 0  # 如果某个维度上相等（same_d_count = self.D_train），则 +1，如果每个维度上的投影都相交，则说明两个 CU 重叠
                for d in range(self.D_train):
                    if self.is_overlapping(point_1_start=start_point_I[d], point_1_end=end_point_I[d],
                                           point_2_start=start_point_S[d], point_2_end=end_point_S[d]):
                        same_d_count += 1
                    else:
                        break

                # 如果 I 与 S 不相邻，直接进行下一次迭代
                if same_d_count != self.D_train:
                    continue

                # 记录被感染者 S 的下标
                arr_index_S.append(i)
                sum_point += self.arrCU_num_point[i]
                sum_V += (self.arrCU_dL[i] ** self.D_train)

            # 【感染】如果相邻，则进行感染，并且重置他们的感染力度和单元类别 >>>>>>>>>>>>>>>>>
            print(f'[INFO] CU_{index_I} 将感染 {arr_index_S}, {len(arr_index_S)}')
            if len(arr_index_S) == 0:  # 可能出现空的情况
                continue

            new_fi = sum_point / sum_V  # 新感染力度
            for i in arr_index_S:
                self.arrCU_force_infection[i] = new_fi
                self.arrCU_final_target[i] = target_I
            # <<<<<<<<<<<<<<<<<< 感染结束

            if self.is_draw_2D and (2 == self.D_train):
                self.draw_2d(color_map=self.color_map, pic_save_path=self.pic_save_path)

    def draw_2d(self, color_map, pic_save_path=None, title='') -> None:
        """
        绘制 2D 图形
        :param color_map:
        :param pic_save_path:
        :return:
        """
        # self.draw_count += 1
        # if self.draw_count > 70 and self.draw_count % 10 != 0:
        #     return
        self.draw_count += 1
        if not (self.draw_count < 80 or self.draw_count % 10 == 0):
            return

        color_map = np.array(color_map)
        if color_map.shape[0] != self.D_train:
            raise ValueError('\n[CUC-ERROR] color_map.shape[0] != self.D_train')

        plt.figure(figsize=(5, 5))

        # 绘制点
        tmp_data = pd.concat([pd.DataFrame(self.X_TRAIN), pd.Series(self.Y_TRAIN)], axis=1)
        tmp_data.columns = ('x', 'y', 'target')
        for target in np.unique(self.Y_TRAIN):
            plt.scatter(tmp_data[tmp_data['target'] == target].values[:, 0],
                        tmp_data[tmp_data['target'] == target].values[:, 1],
                        c=color_map[target], s=5)

        # 绘制编码单元区块
        for i in range(self.arrCU_start_points.shape[0]):
            target = self.arrCU_final_target[i]
            # 如果当前 CU 不是最终的，或者是 disable 的，那么没有绘制的必要
            if (target == self.CUType.NOT_FINAL_CU.value) or (self.arrCU_is_enable[i] is False):
                continue

            start_points = self.arrCU_start_points[i]
            dL = self.arrCU_dL[i]
            t_x_block = [start_points[0], start_points[0] + dL, start_points[0] + dL, start_points[0]]
            t_y_block = [start_points[1], start_points[1],      start_points[1] + dL, start_points[1] + dL]

            # plt.plot(t_x_block, t_y_block, c='black')  # 黑色边框
            if self.CUType.EMPTY_CU.value == target:
                plt.fill(t_x_block, t_y_block, c='grey', alpha=0.2)
            else:
                plt.fill(t_x_block, t_y_block, c=color_map[target], alpha=0.2)

            plt.title(title)
            plt.ylabel('y')
            plt.xlabel('x')
            plt.axis('equal')  # x、y 单位长度等长

        if pic_save_path is not None:
            plt.savefig(f'{pic_save_path}-{self.draw_count}.png')
        plt.show()
        plt.close('all')  # 必须手动释放内存，不然当需要绘制很多图形时会导致内存泄漏

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """编码单元分类器的估计器（estimator）

        Args:
            X (np.ndarray): 特征值
            y (np.ndarray): 目标值

        Raises:
            ValueError: _description_
        """
        # 如果特征值和目标值维度相等
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'[CUC-ERROR] X.shape != y.shape: X.shape is {X.shape[0]}, y.shape is {y.shape[0]}')

        # =================================== 初始化 CUC 配置参数 ===================================
        self.D_train = X.shape[1]
        self.X_TRAIN = copy.deepcopy(X)
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.df_train = pd.concat([pd.DataFrame(self.X_train), pd.Series((self.y_train), name='target')], axis=1)

        self.CU_min = X.min()
        self.CU_max = X.max()

        # 目标值转为数值类型
        self.transfer_LabelEncoder = LabelEncoder()
        self.y_train = np.array(self.transfer_LabelEncoder.fit_transform(y=y), dtype=np.int)
        self.Y_TRAIN = np.array(self.transfer_LabelEncoder.fit_transform(y=y), dtype=np.int)

        # 初始化 CU 相关 ndarray
        self.arrCU_start_points = np.full(shape=(1, self.D_train), fill_value=self.CU_min)  # 将初始化中的 0.0 替换为编码单元的起始点
        self.arrCU_dL = np.array([self.CU_max - self.CU_min])
        self.arrCU_is_enable = np.array([True])
        self.arrCU_final_target = np.array([self.CUType.NOT_FINAL_CU.value], dtype=np.int)
        self.arrCU_force_infection = np.array([np.nan])
        # =========================================================================================


        # ========================================= 预分割 =========================================
        self.pre_split()
        self.remove_disable_points()

        print('\n>>>>>>>>>>>>>>>> 完成预分割后的arr结果')
        res = pd.concat([pd.DataFrame(self.arrCU_start_points),
                         pd.Series(self.arrCU_is_enable),
                         pd.Series(self.arrCU_dL),
                         pd.Series(self.arrCU_final_target),
                         pd.Series(self.arrCU_force_infection)], axis=1)
        res.columns = ('x', 'y', 'is_enable', 'dL', 'target', 'force_infection')
        print(res)
        # =========================================================================================


        # ========================================= 细化分割 =========================================
        self.refinement_split(num=self.num_refinement_splits)
        self.remove_disable_points()
        if self.D_train == 2 and self.is_draw_2D:
            self.draw_2d(color_map=self.color_map, pic_save_path=self.pic_save_path)

        print('\n>>>>>>>>>>>>>>>> 完成细化分割后的arr结果')
        res = pd.concat([pd.DataFrame(self.arrCU_start_points),
                         pd.Series(self.arrCU_is_enable),
                         pd.Series(self.arrCU_dL),
                         pd.Series(self.arrCU_final_target),
                         pd.Series(self.arrCU_force_infection)], axis=1)
        res.columns = ('x', 'y', 'is_enable', 'dL', 'target', 'force_infection')
        print(res)
        # =========================================================================================

        self.infection()

    def predict(self, X: np.ndarray) -> list:
        if X.shape[1] != self.D_train:
            raise ValueError(f'[CUC-ERROR] In method predict: X.shape != self.D_train: X.shape is {X.shape[1]}, y.shape is {self.D_train}')

        # 根据 point 对应的编码单元预测结果
        res_predict = []
        for x in X:
            cur_len = len(res_predict)
            for i in range(self.arrCU_start_points.shape[0]):
                if self.predict_is_point_in_CU(point=x, start_point=self.arrCU_start_points[i], dL=self.arrCU_dL[i]):
                    res_predict.append(self.arrCU_final_target[i])
                    break

            if cur_len == len(res_predict):  # 有的点可能不落在任意一 CU 里，给他一个随机值
                res_predict.append(np.random.randint(low=0,
                                                     high=self.transfer_LabelEncoder.classes_.shape[0]))

        return self.transfer_LabelEncoder.inverse_transform(y=res_predict)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if X.shape[1] != self.D_train:
            raise ValueError(f'[CUC-ERROR] In method predict: X.shape != self.D_train: X.shape is {X.shape[1]}, y.shape is {self.D_train}')

        # 根据 point 对应的编码单元预测结果
        res_predict = []
        for x in X:
            cur_len = len(res_predict)
            for i in range(self.arrCU_start_points.shape[0]):
                if self.predict_is_point_in_CU(point=x, start_point=self.arrCU_start_points[i], dL=self.arrCU_dL[i]):
                    res_predict.append(self.arrCU_final_target[i])
                    break

            if cur_len == len(res_predict):  # 有的点可能不落在任意一 CU 里，给他一个随机值
                res_predict.append(np.random.randint(low=0,
                                                     high=self.transfer_LabelEncoder.classes_.shape[0]))

        # 将 y 转换为数字类型，因为 y 可能是字符串或者其他类型，对这些值检测可能会很慢或出现故障
        series_y = pd.Series(self.transfer_LabelEncoder.transform(y))
        if len(res_predict) != series_y.shape[0]:
            raise ValueError(
                f'[CUC-ERROR] Shape res_predict shape != y.shape:'
                f' res_predict shape is {len(res_predict)}, y.shape is {series_y.shape[0]}')

        # 拼接预测试和真实值，以便于后期处理
        df_res = pd.concat([pd.Series(res_predict), series_y], axis=1)
        df_res.columns = ('predict', 'target')

        correct_num = df_res.query('predict == target').count()[0]  # 预测正确的结果数量
        return correct_num / series_y.shape[0]  # 计算正确率并返回
