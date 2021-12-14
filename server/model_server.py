import fasttext
import os
import pandas as pd
import time
from pathlib import Path
from preprocess_text import preprocess_text
from deploy_model import deploy_and_run

data_path = 'data/'


def train_model():
    while True:
        if not os.path.isfile(data_path + 'data_new.csv'):
            continue

        time.sleep(1)

        df_new = pd.read_csv(data_path + 'data_new.csv', index_col=0)

        if len(df_new) < 5000:
            continue

        print("Starting to train...")

        df_old = pd.read_csv(data_path + 'data_old.csv', index_col=0)
        #df_test = pd.read_csv(data_path + 'data_test.csv', index_col=0)
        df_new_sample = df_new.sample(n=5000)

        df_new = df_new.drop(df_new_sample.index)
        if len(df_new):
            df_new.to_csv(data_path + "data_new.csv")
        else:
            os.remove(data_path + 'data_new.csv')

        df_train = pd.concat([df_new_sample, df_old.sample(n=5000)], 
                             ignore_index=False)
        df_train['text'] = df_train['text'].apply(preprocess_text)

        with open('fasttext_train.txt', 'w') as f:
            for each_text, each_label in zip(df_train['text'], df_train['label']):
                f.writelines(f'__label__{each_label} {each_text.encode("utf-8")}\n')

        model = fasttext.train_supervised("fasttext_train.txt", lr=0.5,
                                          epoch=5, wordNgrams=2, dim=50,
                                          loss='softmax', verbose=1)

        print("Training finished.")

        accuracy = model.test('fasttext_test.txt')[1]

        print("Accuracy: ", accuracy)

        if accuracy < 0.7:
            print("WARNING: THE ACCURACY IS BAD ! The previous model is reloaded.")
            continue

        df_old = pd.concat([df_new_sample, df_old], ignore_index=False)
        df_old.to_csv(data_path + "data_old.csv")

        model.save_model(str(Path(__file__).parent.joinpath(Path("docker/model_fasttext.bin"))))
        deploy_and_run()

if __name__ == '__main__':
    train_model()

