import json
import os

from flask import jsonify
from keras.models import load_model
import numpy as np
from google.cloud import storage
from keras import backend as K


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# make prediction
def pre_predict(test_df):
    sequence_length = 50
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)
    seq_array_test_last = [test_df[test_df['id'] == id][sequence_cols].values[-sequence_length:]
                           for id in test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= sequence_length]

    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    y_mask = [len(test_df[test_df['id'] == id]) >= sequence_length for id in test_df['id'].unique()]
    label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)
    return label_array_test_last, seq_array_test_last


def predict(result):
    project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
    model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
    if model_repo:
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(model_repo)
        blob = bucket.blob('model.h5')
        blob.download_to_filename('local_model.h5')
        model = load_model('local_model.h5')

        y_pred_test = model.predict(result[1])
        y_true_test = result[0]
        y_classes = y_pred_test.argmax(axis=-1)
        y = {"y_pred": y_pred_test.tolist(), "y_true": y_true_test.tolist(), "y_pred_classes": y_classes.tolist()}
        return json.dumps(y)
    else:
        return jsonify({'message': 'MODEL_REPO cannot be found.'}), 200


def show_metrics(result):
    result = json.loads(result)
    precision_test = str(precision_m(result['y_true'], result['y_pred']).numpy())
    recall_test = str(recall_m(result['y_true'], result['y_pred']).numpy())
    f1_test = str(f1_m(result['y_true'], result['y_pred']).numpy())
    text_out = {
        "Precision:": precision_test,
        "Recall": recall_test,
        "F1-score": f1_test
    }
    return jsonify(text_out), 200
