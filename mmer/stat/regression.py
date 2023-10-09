        
import numpy as np
import os
import pandas as pd
import statsmodels.formula.api as smf
from researchtoolbox.utility import preprocessing as pre

"""
This module contains classes and methods for transforming a BVP signal in a BPM signal.
"""

class LinearRegression:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    
    def get_data_frame_based_on_residual_criteria(self, df_threshold, column_name_1, column_name_2):
        """
        Filter out ourlier and return a new data frame using inter-quartile range rule.
        """
        df_residual = pd.DataFrame()
        #df_residual["rPPG_SDNN"] = df_threshold["SDNN"]  # x axis
        #df_residual["rPPG_rMSSD"] = df_threshold["rMSSD"]  # y axis
        df_residual["rPPG_SDNN"] = df_threshold[column_name_1]  # x axis
        df_residual["rPPG_rMSSD"] = df_threshold[column_name_2]  # y axis
        df_residual.dropna(axis=0, inplace=True)
        result = smf.ols('rPPG_rMSSD~rPPG_SDNN', data=df_residual).fit()  # y~x
        intercept = result.params.Intercept
        theta = result.params.rPPG_SDNN
        y_pred = intercept + theta * df_residual["rPPG_SDNN"]
        df_residual["residual"] = (df_residual["rPPG_rMSSD"] - y_pred).abs()

        df_residual, _ = pre.Outlier().get_outliers(df_residual, ["residual"])

        # k = 1.5
        # upper_bound = np.percentile(df_residual["residual"], 75) + k * (
        #         np.percentile(df_residual["residual"], 75) - np.percentile(df_residual["residual"], 25))
        # lower_bound = np.percentile(df_residual["residual"], 25) - k * (
        #         np.percentile(df_residual["residual"], 75) - np.percentile(df_residual["residual"], 25))
        # df_residual = df_residual[df_residual["residual"] < upper_bound]
        # df_residual = df_residual[df_residual["residual"] >= lower_bound]
        # print("upper bound: ", upper_bound)
        return df_residual
