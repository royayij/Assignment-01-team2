import os

from flask import jsonify
from keras.models import load_model
import numpy as np
from sklearn.metrics import precision_score, recall_score


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
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path = os.path.join(model_repo, "model.h5")
        model = load_model(file_path)
        # scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose=2)

        y_pred_test = model.predict(seq_array_test_last)
        y_classes = y_pred_test.argmax(axis=-1)
        y_true_test = label_array_test_last
        precision_test = precision_score(y_true_test, y_classes)
        recall_test = recall_score(y_true_test, y_classes)
        f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
        text_out = {
            "Precision:": precision_test,
            "Recall": recall_test,
            "F1-score": f1_test
        }
        return jsonify(text_out), 200
    else:
        return jsonify({'message': 'MODEL_REPO cannot be found.'}), 200
