import re
#import spacy
#import en_core_web_sm
import fasttext
import flask
import random
import json
from multiprocessing import Process, Value
import os
import pandas as pd

#nlp = en_core_web_sm.load()
api = flask.Flask(__name__)
data_path = 'data/'
has_to_load = Value('d', 0)
model = None

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

def train_model(has_to_load):
    #if __name__ == "__main__":
    #    return
    while True:
        if not os.path.isfile(data_path + 'data_new.csv'):
            continue

        df_new = pd.read_csv(data_path + 'data_new.csv', index_col=0)

        if len(df_new) < 3000:
            continue
        print('module name:', __name__)
        print('parent process:', os.getppid())
        print('process id:', os.getpid())

        print("Starting to train...")

        df_old = pd.read_csv(data_path + 'data_old.csv', index_col=0)
        df_test = pd.read_csv(data_path + 'data_test.csv', index_col=0)
        df_new_sample = df_new.sample(n=3000)


        df_train = pd.concat([df_new_sample, df_old.sample(n=3000),
                              df_test ], ignore_index=False)
        df_train['text'] = df_train['text'].apply(preprocess_text)

        df_new = df_new.drop(df_new_sample.index)
        df_new.to_csv(data_path + "data_new.csv")

        df_old = pd.concat([df_new_sample, df_old], ignore_index=False)
        df_new.to_csv(data_path + "data_old.csv")

        with open('fasttext_train.txt', 'w') as f:
            for each_text, each_label in zip(df_train['text'], df_train['label']):
                f.writelines(f'__label__{each_label} {each_text}\n')

        model = fasttext.train_supervised("fasttext_train.txt", lr=0.5,
                                          epoch=5, wordNgrams=2, dim=50,
                                          loss='softmax', verbose=1)
       # os.remove('fasttext_train.txt')

        model.save_model("model_fasttext.bin")

        print("Training finished.")

        has_to_load.value = 1



@api.route('/predict', methods=['POST'])
def predict():
    print(has_to_load.value)
    if has_to_load.value:
        has_to_load.value = 0
        model = fasttext.load_model("model_fasttext.bin")
    if not model:
        print("Model is not ready.")
        return ""
    text = request.args.get('text')
    label, score = model.predict(preprocess_text(text))

    response = json.dumps(
        {
            "label": label[0][-1],
            "score": score[0]
        }
    )
    return response

@api.route('/')
def hello():
    return "Prediction REST API"

if __name__ == '__main__':
    p_train_model = Process(target=train_model, args=(has_to_load,))
    p_train_model.start()
    api.run(debug=True)

