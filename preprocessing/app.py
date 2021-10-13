from flask import Flask, Response, json
import os

import pandas as pd
import requests

from resources.preprocessing import pre_process_train_db, pre_process_test_db

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/preprocessing/<table_name>', methods=['GET'])
def preprocess_db(table_name):
    if table_name == 'train_db':
        try:
            db_api = os.environ['TRAININGDB_API']
        except:
            return json.dumps({'message': 'train set is not imported'}, sort_keys=False, indent=4), 200
    elif table_name == 'test_db':
        try:
            db_api = os.environ['TESTDB_API']
        except:
            return json.dumps({'message': 'test set is not imported'}, sort_keys=False, indent=4), 200
    else:
        return json.dumps({'message': 'the name of dataset is not correct'}, sort_keys=False, indent=4), 200
    r = requests.get(db_api)
    j = r.json()
    df = pd.DataFrame.from_dict(j)
    if table_name == 'train_db':
        pre_df = pre_process_train_db(df)
    else:
        pre_df = pre_process_test_db(df)
    resp = Response(pre_df.to_json(orient='records'), status=200, mimetype='application/json')
    return resp


app.run(host='0.0.0.0', port=5000)
