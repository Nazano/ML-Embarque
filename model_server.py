import re
#import spacy
#import en_core_web_sm
import fasttext
import flask
import random
import json
from multiprocessing import Process, Value, Pool
import os
import pandas as pd
import time

#nlp = en_core_web_sm.load()
api = flask.Flask(__name__)
data_path = 'data/'
has_to_load = Value('d', 0)

def preprocess_text(text):
    '''
    Returns a list of words of a preprocessed document
        Parameter:
            text(str): The document
        Return:
            The proprocessed text
    '''
    # Remove delimiters
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # Remove all the special characters
    text = re.sub(r'\W', ' ', text)
    # Remove numbers
    text = re.sub(r'[0-9]+', ' ', text)
    # Replace multiple space by one
    text = re.sub(r' +', ' ', text)

    # Converting to Lowercase
    text = text.lower()

    return text
    # Lemmatization
    #tokens = [token.lemma_ for token in nlp(text) if token.lemma_]
    # Remove french stop words
    #tokens = [word for word in tokens if word not in fr_stop]

    return " ".join(tokens)

def train_model():
    while True:
        if not os.path.isfile(data_path + 'data_new.csv'):
            continue

        time.sleep(1)

        df_new = pd.read_csv(data_path + 'data_new.csv', index_col=0)

        if len(df_new) < 3000:
            continue

        print("Starting to train...")

        df_old = pd.read_csv(data_path + 'data_old.csv', index_col=0)
        df_test = pd.read_csv(data_path + 'data_test.csv', index_col=0)
        df_new_sample = df_new.sample(n=3000)


        df_train = pd.concat([df_new_sample, df_old.sample(n=3000),
                              df_test ], ignore_index=False)
        df_train['text'] = df_train['text'].apply(preprocess_text)

        df_new = df_new.drop(df_new_sample.index)
        if len(df_new):
            df_new.to_csv(data_path + "data_new.csv")
        else:
            os.remove(data_path + 'data_new.csv')

        df_old = pd.concat([df_new_sample, df_old], ignore_index=False)
        df_old.to_csv(data_path + "data_old.csv")

        with open('fasttext_train.txt', 'w') as f:
            for each_text, each_label in zip(df_train['text'], df_train['label']):
                f.writelines(f'__label__{each_label} {each_text}\n')

        model = fasttext.train_supervised("fasttext_train.txt", lr=0.5,
                                          epoch=5, wordNgrams=2, dim=50,
                                          loss='softmax', verbose=1)
       # os.remove('fasttext_train.txt')

        model.save_model("model_fasttext.bin")

        print("Training finished.")




@api.route('/predict', methods=['POST'])
def predict():
    model = fasttext.load_model("model_fasttext.bin")
    text = flask.request.get_json()['text']
    label, score = model.predict(preprocess_text(text))

    response = json.dumps(
        {
            "label": label[0][-1],
            "score": score[0]
        }
    )
    print(response)
    return response

@api.route('/')
def hello():
    return "Prediction REST API"

#if __name__ == '__main__':
Process(target=train_model).start()
model = fasttext.load_model("model_fasttext.bin")
api.run(debug=False)

