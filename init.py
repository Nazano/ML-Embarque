import os
import pandas as pd

data_path = 'data/'

if os.path.isfile(data_path + 'new_data.csv'):
    os.remove(data_path + "new_data.csv")
df_save = pd.read_csv(data_path + "data_save.csv", index_col=0)

df_test = df_save.sample(n=3000)
df_data = df_save.drop(df_test.index)
df_old = df_data.sample(n=3000)
df_data = df_data.drop(df_old.index)

df_data.to_csv(data_path + 'data.csv')
df_test.to_csv(data_path + 'data_test.csv')
df_old.to_csv(data_path + 'data_old.csv')

