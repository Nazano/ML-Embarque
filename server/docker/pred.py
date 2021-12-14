import flask
import fasttext
import json
import os
import re

api = flask.Flask(__name__)

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


@api.route('/predict', methods=['POST'])
def predict():
    if not os.path.isfile("model_fasttext.bin"):
        return "Model is not ready.", 555
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

if __name__ == "__main__":
    api.run(host='0.0.0.0')