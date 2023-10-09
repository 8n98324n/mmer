import os
import pandas as pd
import numpy as np

from researchtoolbox.stat import regression as reg
from researchtoolbox.utility import preprocessing as pre
from researchtoolbox.constant_variable import *

class PreAnalysis:
    def __int__(self):
        pass

    def process(self, input_file_path, save_path):

        df = pd.read_csv(input_file_path)

        df["ln(HF)"] = df["HF"].apply(np.log)
        df["ln(LF)"] = df["LF"].apply(np.log)
        df["ln(LF/HF)"] = df["LFHF"].apply(np.log)
        r_squre_list = []
        # 对rPPG的数据进行筛选: 保留pNN50>0，并去除residual为outlier的row
        criterion_column = "Valid Ratio"
        df_threshold = df[df["pNN50"] > 0]
        df_residual = reg.LinearRegression().get_data_frame_based_on_residual_criteria(df_threshold, "SDNN", "rMSSD")
        index = np.array(df_residual.index).tolist()
        df_validation = df_threshold.loc[index, :]
        df_validation["residual"] = df_residual["residual"]
        df_validation, _ = pre.Outlier().get_outliers(df_validation, hrv_variable_list)

        # 计算同一个人同一个点的结束/开始
        df = df_validation.copy()
        ref_list = df.loc[:, "No-Step"].tolist()
        df.loc[:, hrv_difference_index] = np.nan
        for row in df.index:
            no_step = "_".join(df.loc[row, "No-Step"].split("_")[:-1])
            if no_step + "_0" in ref_list:
                start = 0
            else:
                start = -1

            temp_list = []
            for ind in range(0, 20):
                if no_step + "_" + str(ind) in ref_list:
                    temp_list.append(ind)
            end = max(temp_list)

            if end > start > -1:  # 说明这个人的数据是可用的（开头是第0段，有结尾，且开头≠结尾）
                end_index = df[df["No-Step"] == no_step + "_" + str(end)].index[0]  # 如果有多个重复行，取第一个的下标
                start_index = df[df["No-Step"] == no_step + "_" + str(start)].index[0]  # 如果有多个重复行，取第一个的下标
                # df.loc[row, hrv_difference_index] = (100*(df.loc[end_index, hrv_variable_index]/df.loc[start_index, hrv_variable_index]).astype('float').apply(np.log)).copy().values.tolist()
                df.loc[row, hrv_difference_index] = (
                            df.loc[end_index, hrv_variable_index] - df.loc[start_index, hrv_variable_index]).astype(
                    'float').copy().values.tolist()  # 此处做减法

        # step 45/50 --> step 46
        df["Step-new"] = df["Step"]
        df["Step-new"].replace(45, 46, inplace=True)
        df["Step-new"].replace(50, 46, inplace=True)

        df.to_csv(save_path)  # 保存
        return df


