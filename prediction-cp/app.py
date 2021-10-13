import pandas as pd
from flask import Flask
import os
import requests

from resources import predictor

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/prediction-cp/results', methods=['POST'])
def predict_perf():
    # receive the prediction request data as the message body
    db_api = os.environ['PREPROCESSDB_API']
    # Make a GET request to training db service to retrieve the training data/features.
    r = requests.get(db_api)
    j = r.json()
    df = pd.DataFrame.from_dict(j)
    resp = predictor.predict(df)
    return resp


app.run(host='0.0.0.0', port=5000)
