import os

import pandas as pd
import requests
from flask import Flask, json

from flask import jsonify
from resources import model_trainer

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/vizualization-cp/figures', methods=['POST'])
def make_figure():
    db_api = os.environ['PREDICTIONDB_API']
    # Make a GET request to training db service to retrieve the training data/features.
    r = requests.get(db_api)
    j = r.json()
    df = pd.DataFrame.from_dict(j)
    resp = figure_maker.make_fig(df)
    return resp


app.run(host='0.0.0.0', port=5000)
