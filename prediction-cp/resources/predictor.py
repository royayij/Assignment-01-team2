import os

from flask import jsonify
from keras.models import load_model
import numpy as np
from sklearn.metrics import precision_score, recall_score
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
def predict(test_df):
    sequence_length = 50
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    seq_array_test_last = [test_df[test_df['id'] == id][sequence_cols].values[-sequence_length:]
                           for id in test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= sequence_length]

    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    y_mask = [len(test_df[test_df['id'] == id]) >= sequence_length for id in test_df['id'].unique()]
    label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)
    project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
    model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
    if model_repo:
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(model_repo)
        blob = bucket.blob('model.h5')
        blob.download_to_filename('local_model.h5')
        model = load_model('local_model.h5')

        # scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose=2)

        y_pred_test = model.predict(seq_array_test_last)
        y_true_test = label_array_test_last
        precision_test = precision_m(y_true_test, y_pred_test).numpy()
        recall_test = recall_m(y_true_test, y_pred_test).numpy()
        f1_test = f1_m(y_true_test, y_pred_test).numpy()
        text_out = {
            "Precision:": precision_test,
            "Recall": recall_test,
            "F1-score": f1_test
        }
        return jsonify(text_out), 200
    else:
        return jsonify({'message': 'MODEL_REPO cannot be found.'}), 200
