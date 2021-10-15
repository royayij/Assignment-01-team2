import pandas as pd
from flask import Flask, Response
import os
import requests
import json

from resources import predictor

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/prediction-cp/metrics', methods=['POST'])
def predict_metric():
    # receive the prediction request data as the message body
    db_api = os.environ['PREPROCESSDB_API']
    # Make a GET request to preprocess db service to retrieve the preprocess data/features.
    r = requests.get(db_api)
    j = r.json()
    df = pd.DataFrame.from_dict(j)
    pre_result = predictor.pre_predict(df)
    result = predictor.predict(pre_result)
    resp = predictor.show_metrics(result)
    return resp


@app.route('/prediction-cp/results', methods=['GET'])
def predict_results():
    # receive the prediction request data as the message body
    db_api = os.environ['PREPROCESSDB_API']
    # Make a GET request to preprocess db service to retrieve the preprocess data/features.
    r = requests.get(db_api)
    j = r.json()
    df = pd.DataFrame.from_dict(j)
    pre_result = predictor.pre_predict(df)
    result = predictor.predict(pre_result)
    resp = Response(json.dumps(result), status=200, mimetype='application/json')

    return resp


app.run(host='0.0.0.0', port=5000)
