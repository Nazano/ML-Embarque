import flask
import random
import json

api = flask.Flask(__name__)

@api.route('/predict', methods=['POST'])
def predict():
    response = json.dumps(
        {
            "label": random.randint(0, 1),
            "score": random.random()
        }
    )
    return response

@api.route('/')
def hello():
    return "Prediction REST API"

if __name__ == '__main__':
    api.run(debug=True)