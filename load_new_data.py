#!/usr/bin/python

import sys
import pandas as pd
import os

data_path = 'data/'

if __name__ == '__main__':
    n = 3000 if len(sys.argv) != 2 else int(sys.argv[1])
    df_data = pd.read_csv(data_path + "data.csv", index_col=0)
    df_sample = df_data.sample(n=n)
    df_data = df_data.drop(df_sample.index)
    df_data.to_csv(data_path + "data.csv")

    if not os.path.isfile(data_path + 'data_new.csv'):
        df_new = df_sample
    else:
        df_new = pd.concat([pd.read_csv(data_path + "data_new.csv", index_col=0),
                            df_sample], 
                           ignore_index=False)
    df_new.to_csv(data_path + "data_new.csv")
