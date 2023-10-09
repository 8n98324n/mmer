import pandas as pd


class Dataset:
    def __init__(self):
        pass

    def read_dataset(self, columns, data_path, step):
        df = pd.read_csv(data_path)
        df = df.loc[df["Step-new"].isin(step)]

        df_data = df.loc[:, columns]
        df_data.dropna(axis=0, how='all', inplace=True)  # 删掉全是空值的行

        x = df_data.fillna(df_data.median())  # 插值法
        y = df.loc[x.index, ["Step-new"]]
        remember_no = df.loc[x.index, ["No-Step"]]

        return x, y, remember_no