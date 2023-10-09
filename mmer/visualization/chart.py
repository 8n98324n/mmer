#Commonly Used Packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
from researchtoolbox.constant_variable import *


plt.rcParams['axes.unicode_minus'] = False    # 显示负号
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)


"""
This module contains classes and methods for transforming a BVP signal in a BPM signal.
"""

class ScatterPlot:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    def __init__(self):
        pass




class HeatPlot:

    # hrv_variable_list = ["HR", "SDNN", "rMSSD", "pNN50", "ln(HF)", "ln(LF)", "ln(LF/HF)"]

    def __init__(self, data, group1, group2):
        self.exp_group = group1
        self.con_group = group2
        self.data = data


    def single_ROI(self):
        single_ROI_t_test = {}
        value_list = []
        color_list = []
        color_21_list = []
        color_11_list = []
        value_21_list = []
        value_11_list = []

        for i in hrv_difference_index:
            exp_group_list = self.data[self.data["Step"] == self.exp_group][i].dropna().tolist()
            con_group_list = self.data[self.data["Step"] == self.con_group][i].dropna().tolist()

            single_ROI_t_test[i] = ttest_ind(exp_group_list, con_group_list)
            # single_ROI_t_test[i] = ttest_rel(exp_group_list, con_group_list)  # 组内t-test
            avg_diff = np.mean(exp_group_list) - np.mean(con_group_list)

            value_21_list.append(np.mean(exp_group_list))
            value_11_list.append(np.mean(con_group_list))
            value_list.append(avg_diff)
            color_21_list.append(ttest_1samp(exp_group_list, 0).pvalue)
            color_11_list.append(ttest_1samp(con_group_list, 0).pvalue)
            color_list.append(single_ROI_t_test[i].pvalue)
            # color_list.append(rel_t_test_results[i].pvalue)  # 组内t-test
            print("{}\t{}\t{}".format(i, round(avg_diff, 3), round(single_ROI_t_test[i].pvalue, 3)))

        value_array = np.array([value_21_list, value_11_list, value_list])
        color_array = np.array([color_21_list, color_11_list, color_list])

        print()
        print(value_array)
        print(color_array)

        plt.xticks(np.arange(len(hrv_variable_list)), labels=hrv_variable_list, fontsize=12)
        plt.yticks(np.arange(3),
                   labels=[str(self.exp_group) + "_mean", str(self.con_group) + "_mean", "{}-{} difference".format(self.exp_group, self.con_group)],
                   fontsize=12)

        for i in range(3):
            for j in range(len(hrv_variable_list)):
                if color_array[i, j] >= 0.05:
                    color = "black"
                else:
                    color = "white"
                text = plt.text(j, i, round(value_array[i, j], 2), ha="center", va="center", color=color, fontsize=10)

        top = mpl.colormaps['YlOrRd_r']
        bottom = mpl.colormaps['YlGn_r']  # YlGn_r
        newcolors = np.vstack((top(np.linspace(0, 1, 5 * 5 * 2)[: 25]),
                               bottom(np.linspace(0, 1, 95 * 5 * 2)[95 * 5:])))
        newcmp = ListedColormap(newcolors, name="OrangeBlue")
        plt.imshow(color_array, vmin=0, vmax=1, cmap=newcmp)
        plt.tight_layout()

        plt.savefig("single_ROI.png", dpi=450)
        plt.show()

    def single_ROI_thermal(self):
        single_ROI_t_test = {}
        rel_t_test_results = {}
        value_list = []
        color_list = []
        color_21_list = []
        color_11_list = []
        value_21_list = []
        value_11_list = []

        # 重复测量t-test（组内检验）
        # df_rel_t_test = pd.merge(self.data[self.data["Step-new"] == group_1].loc[:, ['No'] + point_difference_index],
        #                          self.data[self.data["Step-new"] == group_2].loc[:, ['No'] + point_difference_index],
        #                          how="inner", on="No")
        # df_rel_t_test.drop_duplicates(['No'], inplace=True, keep='first')
        # df_rel_t_test.dropna(inplace=True)
        # for i in index_list:
        #     rel_t_test_results["{}_difference".format(i)] = ttest_rel(df_rel_t_test["{}_difference_x".format(i)],
        #                                                             df_rel_t_test["{}_difference_y".format(i)])

        for i in thermal_selected_index_list:
            exp_group_list = self.data[self.data["Step-new"] == self.exp_group]["{}_difference".format(i)].dropna().tolist()
            con_group_list = self.data[self.data["Step-new"] == self.con_group]["{}_difference".format(i)].dropna().tolist()

            single_ROI_t_test["{}_difference".format(i)] = ttest_ind(exp_group_list, con_group_list)
            # single_ROI_t_test["{}_difference".format(i)] = ttest_rel(exp_group_list, con_group_list)
            avg_diff = np.mean(exp_group_list) - np.mean(con_group_list)

            value_21_list.append(np.mean(exp_group_list))
            value_11_list.append(np.mean(con_group_list))
            value_list.append(avg_diff)
            color_21_list.append(ttest_1samp(exp_group_list, 0).pvalue)
            color_11_list.append(ttest_1samp(con_group_list, 0).pvalue)
            color_list.append(single_ROI_t_test["{}_difference".format(i)].pvalue)
            # color_list.append(rel_t_test_results["{}_difference".format(i)].pvalue)
            # print("{}_difference\t{}\t{}".format(i, round(avg_diff, 3), round(single_ROI_t_test["{}_difference".format(i)].pvalue, 3)))

        # value_array = np.array([value_21_list, value_11_list, value_list])
        value_array = np.array([color_21_list, color_11_list, color_list])
        color_array = np.array([color_21_list, color_11_list, color_list])

        plt.xticks(np.arange(len(thermal_selected_index_list)), labels=thermal_selected_index_list, fontsize=8)
        plt.yticks(np.arange(3), labels=[str(self.exp_group) + "_mean", str(self.con_group) + "_mean",
                                         "{}-{} difference".format(self.exp_group, self.con_group)], fontsize=8)

        for i in range(3):
            for j in range(len(thermal_selected_index_list)):
                if color_array[i, j] >= 0.05:
                    color = "black"
                else:
                    color = "white"
                text = plt.text(j, i, round(value_array[i, j], 3), ha="center", va="center", color=color,
                                fontsize=4)

        top = mpl.colormaps['YlOrRd_r']
        bottom = mpl.colormaps['YlGn_r']  # YlGn_r
        newcolors = np.vstack((top(np.linspace(0, 1, 5 * 5 * 2)[: 25]),
                               bottom(np.linspace(0, 1, 95 * 5 * 2)[95 * 5:])))
        newcmp = ListedColormap(newcolors, name="OrangeBlue")
        plt.imshow(color_array, vmin=0, vmax=1, cmap=newcmp)
        # plt.colorbar()
        plt.tight_layout()

        plt.savefig("single_ROI.png", dpi=450)
        plt.show()

    def draw_heatmap(self):
        t_test_results = {}
        rel_t_test_results = {}
        diff_diff_list = []
        # 点与点之间的处理
        for i in thermal_selected_index_list:
            for j in thermal_selected_index_list:
                self.data.loc[:, "{}-{}_diff".format(i, j)] = self.data.loc[:, "{}_difference".format(i)] - self.data.loc[:, "{}_difference".format(j)].copy().values.tolist()
                # self.data.loc[:, "{}-{}_diff".format(i, j)] = (100*(self.data.loc[:, i]/self.data.loc[:, "{}_difference".format(j)]).apply(np.log10)).copy().values.tolist()
                diff_diff_list.append("{}-{}_diff".format(i, j))
                # exp_group_list = self.data[self.data["Step-new"] == self.exp_group]["{}-{}_diff".format(i, j)].dropna().tolist()
                # con_group_list = self.data[self.data["Step-new"] == self.con_group]["{}-{}_diff".format(i, j)].dropna().tolist()
                exp_group_list = self.data[self.data["Step-new"] == self.exp_group]["{}_difference".format(i)].dropna().tolist()
                con_group_list = self.data[self.data["Step-new"] == self.exp_group]["{}_difference".format(j)].dropna().tolist()
                t_test_results["{}-{}_diff".format(i, j)] = ttest_ind(exp_group_list, con_group_list)

        # df_rel_t_test = pd.merge(self.data[self.data["Step-new"] == self.exp_group].loc[:, ['No'] + diff_diff_list],
        #                          self.data[self.data["Step-new"] == self.con_group].loc[:, ['No'] + diff_diff_list], how="inner",
        #                          on="No")
        # df_rel_t_test.drop_duplicates(['No'], inplace=True, keep='first')
        # df_rel_t_test.dropna(inplace=True)
        # df_rel_t_test.to_excel("t_test_rel_thermal_data.xlsx", index=False)
        # self.data.to_excel("ml_thermal_data.xlsx", index=False)

        # for i in thermal_selected_index_list:
        #     for j in thermal_selected_index_list:
        #         rel_t_test_results["{}-{}_diff".format(i, j)] = ttest_rel(df_rel_t_test["{}-{}_diff_x".format(i, j)],
        #                                                                   df_rel_t_test["{}-{}_diff_y".format(i, j)])

        df_group = self.data.groupby("Step-new")
        df_new = df_group[point_difference_index].agg("mean").T
        # print(df_new.columns)  # 11.0, 21.0, 42.0, 46.0
        # df_new["step diff 21-11"] = df_new[self.exp_group]-df_new[self.con_group]
        df_new["step diff group1-group2"] = df_new[self.exp_group]-df_new[self.con_group]

        column_name = self.exp_group  # 呈现在热力图的value值的列

        heatmap_color_data = []  # p-value
        heatmap_value_data = []

        for i in range(len(thermal_selected_index_list)):  # 第i行
            column_color_data = []
            column_value_data = []
            for j in range(len(thermal_selected_index_list)):  # 第j列， 热力图：i/j
                # p_value = df_new.loc["{}_difference".format(thermal_selected_index_list[i]), column_name]/df_new.loc["{}_difference".format(thermal_selected_index_list[j]), column_name]
                p_value = t_test_results["{}-{}_diff".format(thermal_selected_index_list[j], thermal_selected_index_list[i])].pvalue
                # p_value = rel_t_test_results["{}-{}_diff".format(thermal_selected_index_list[j], thermal_selected_index_list[i])].pvalue
                value = df_new.loc["{}_difference".format(thermal_selected_index_list[j]), column_name] - df_new.loc[
                    "{}_difference".format(thermal_selected_index_list[i]), column_name]
                column_value_data.append(value)
                column_color_data.append(p_value)
            heatmap_color_data.append(column_color_data)
            heatmap_value_data.append(column_value_data)

        # # seaborn 画热力图，不太好看
        # heatmap_array = np.array(heatmap_color_data)
        # output = pd.DataFrame(heatmap_array, index=thermal_selected_index_list, columns=thermal_selected_index_list)
        # output.to_excel("output.xlsx")
        # ax = sns.heatmap(heatmap_array, xticklabels=thermal_selected_index_list, yticklabels=index_list, annot=True, fmt="0.1g", linewidths=.5,
        #                  annot_kws={'size': 6}, vmin=0, vmax=0.05, cmap='YlGnBu_r')
        # plt.show()

        # plt 绘制热力图
        heatmap_color_array = np.array(heatmap_color_data)
        heatmap_value_array = np.array(heatmap_value_data)
        # heatmap_value_array = np.array(heatmap_color_data)
        # plt.xticks(np.arange(len(thermal_selected_index_list)), labels=thermal_selected_index_list, rotation=45, rotation_mode="anchor", ha="right")
        plt.xticks(np.arange(len(thermal_selected_index_list)), labels=thermal_selected_index_list, fontsize=8)
        plt.yticks(np.arange(len(thermal_selected_index_list)), labels=thermal_selected_index_list, fontsize=8)
        for i in range(len(thermal_selected_index_list)):
            for j in range(len(thermal_selected_index_list)):
                if heatmap_color_array[i, j] >= 0.05:
                    color = "black"
                else:
                    color = "white"
                text = plt.text(j, i, round(heatmap_value_array[i, j], 2), ha="center", va="center", color=color,
                                fontsize=4)
        top = mpl.colormaps['YlOrRd_r']
        bottom = mpl.colormaps['YlGn_r']  # YlGn_r
        newcolors = np.vstack((top(np.linspace(0, 1, 5 * 5 * 2)[: 25]),
                               bottom(np.linspace(0, 1, 95 * 5 * 2)[95 * 5:])))
        newcmp = ListedColormap(newcolors, name="OrangeBlue")
        plt.imshow(heatmap_color_array, vmin=0, vmax=1, cmap=newcmp)  # 'YlOrRd_r'
        plt.colorbar()
        plt.tight_layout()

        # 绘制白边
        # x, y = np.meshgrid(np.arange(len(heatmap_color_array)), np.arange(len(heatmap_color_array)))
        # m = np.c_[x[heatmap_color_array.astype(bool)], y[heatmap_color_array.astype(bool)]]
        # for pos in m:
        #     r = plt.Rectangle(pos-0.5, 1, 1, facecolor="none",  edgecolor="w", linewidth=0.8)
        #     plt.gca().add_patch(r)

        plt.savefig("heatmap.png", dpi=450)
        plt.show()
        return df_new
