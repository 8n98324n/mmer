        
import numpy as np
import os
import pandas as pd
from researchtoolbox.constant_variable import *

"""
This module contains classes and methods for transforming a BVP signal in a BPM signal.
"""

class Outlier:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    def __init__(self):
        pass

    def get_outliers(self, df, column_list, k=1.5, RR=False):  # 剔除异常值（1.5(Q3-Q1)）
        outlier_index_list = []
        mask_combined = []
        for column in column_list:
            if RR:
                column_replace = column.replace('std', 'mean')
                df[column_replace].loc[df[column] == 0] = np.nan

            # df[column].loc[df[column] == 0] = np.nan
            mins = df[column] < df.describe()[column]['25%']-k*(df.describe()[column]['75%']-df.describe()[column]['25%'])
            maxs = df[column] > df.describe()[column]['75%']+k*(df.describe()[column]['75%']-df.describe()[column]['25%'])
            mask = mins | maxs

            if RR:
                df.loc[mask, column] = np.nan
                df.loc[mask, column_replace] = np.nan
            else:
                df.loc[mask, column] = np.nan
            outlier_index_list += df.loc[mask, column].index.tolist()
        return df, list(set(outlier_index_list))

class Match:
    def __init__(self, data):
        self.data = data

    def compute_ratio(self):
        df_point = self.data[self.data["use"] == 1]  # 这句其实已经没什么用处了
        df_point = df_point.loc[:, point_RR_index + point_mean_index]

        # 根据每个点的RR单独去掉outlier
        df_point_outlier, _ = Outlier().get_outliers(df_point, point_RR_index, RR=True, k=1.5)
        df_point_outlier, _ = Outlier().get_outliers(df_point_outlier, point_mean_index, RR=False, k=1.5)
        # df_point_outlier = get_outliers(df_point_outlier, point_RR_index, RR=True, k=1.5)

        # 将结果更新到原DataFrame当中
        self.data.loc[:, point_RR_index + point_mean_index] = np.nan  # 先清空相应column
        self.data.loc[:, point_RR_index + point_mean_index] = df_point_outlier
        useful_index = df_point.index
        self.data.loc[useful_index, point_difference_index] = np.nan  # 先清空column，再填充
        self.match_start_end(useful_index)
        return self.data

    def match_start_end(self, useful_index=None):
        # 计算同一个人同一个点的结束/开始
        if useful_index is None:
            useful_index = self.data.index
        ref_list = self.data.loc[useful_index, "No-Step"].tolist()

        for row in useful_index:
            no_step = "-".join(self.data.loc[row, "No-Step"].split("-")[:-1])

            if no_step + "-0" in ref_list:
                start = 0
            else:
                start = -1

            temp_list = []
            for ind in range(0, 20):
                if no_step + "-" + str(ind) in ref_list:
                    temp_list.append(ind)
            end = max(temp_list)

            if end > start > -1:  # 说明有戏
                end_index = self.data[self.data["No-Step"] == no_step + "-" + str(end)].index[0]  # 如果有多个重复行，取第一个的下标
                start_index = self.data[self.data["No-Step"] == no_step + "-" + str(start)].index[0]  # 如果有多个重复行，取第一个的下标
                # df.loc[row, point_difference_index] = (100*(df.loc[end_index, point_mean_index]/df.loc[start_index, point_mean_index]).astype('float').apply(np.log10)).copy().values.tolist()
                self.data.loc[row, point_difference_index] = (self.data.loc[end_index, point_mean_index] -
                                                       self.data.loc[start_index, point_mean_index]).copy().values.tolist()
                self.data.loc[row, point_start_index] = self.data.loc[start_index, point_mean_index].copy().values.tolist()
                self.data.loc[row, point_end_index] = self.data.loc[end_index, point_mean_index].copy().values.tolist()

