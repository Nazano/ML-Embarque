import os
import pandas as pd
from preprocess_text import preprocess_text

data_path = 'data/'

if os.path.isfile(data_path + 'data_new.csv'):
    os.remove(data_path + "data_new.csv")
df_save = pd.read_csv(data_path + "data_save.csv", index_col=0)

df_test = df_save.sample(n=5000)
df_data = df_save.drop(df_test.index)
df_old = df_data.sample(n=5000)
df_data = df_data.drop(df_old.index)

df_data.to_csv(data_path + 'data.csv')
df_test.to_csv(data_path + 'data_test.csv')
df_old.to_csv(data_path + 'data_old.csv')

with open('fasttext_test.txt', 'w') as f:
    for each_text, each_label in zip(df_test['text'].apply(preprocess_text), df_test['label']):
        f.writelines(f'__label__{each_label} {each_text}\n')

