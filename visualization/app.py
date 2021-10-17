from flask import Flask, render_template, url_for
import os
import requests
import pandas as pd
from resources import figure_maker

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/vizualization-cp/acc-fig', methods=['GET'])
def figure_acc():
    figure_maker.acc_fig()
    return render_template('acc_result.html')



@app.route('/vizualization-cp/result-fig', methods=['GET'])
def figure_result():
    db_api = os.environ['PREDICT_API']
    # Make a GET request to training db service to retrieve the training data/features.
    r = requests.get(db_api)
    j = r.json()
    df = pd.DataFrame.from_dict(j)
    figure_maker.result_fig(df)
    return render_template('predict.html')


app.run(host='0.0.0.0', port=5000)
