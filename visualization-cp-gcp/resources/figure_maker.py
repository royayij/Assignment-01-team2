# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import matplotlib.pyplot as plt
import time
from google.cloud import storage
import os
from flask import jsonify
import numpy as np


def acc_fig():
    # fig_verify = plt.figure(figsize=(100, 50))
    # plt.plot(y[0], color="blue")
    # plt.plot(y[1], color="green")
    # plt.title('prediction')
    # plt.ylabel('value')
    # plt.xlabel('row')
    # plt.legend(['predicted', 'actual data'], loc='upper left')
    now = time.time()
    project_id = os.environ['PROJECT_ID']
    figure_repo = os.environ['FIGURE_REPO']
    history_repo = os.environ['HISTORY_REPO']
    fig_acc = plt.figure(figsize=(10, 10))
    if history_repo:
        client = storage.Client(project=project_id)
        history_bucket = client.get_bucket(history_repo)
        history_blob = history_bucket.blob('my_history.npy')
        history_blob.download_to_filename('my_history.npy')
        history = np.load('my_history.npy', allow_pickle='TRUE').item()
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # Save the model localy
        if figure_repo:
            fig_acc.savefig(f"model_acc_{now}.png")
            # Save to GCS
            fig_bucket = client.get_bucket(figure_repo)
            fig_blob = fig_bucket.blob(f"model_verify_{now}.png")
            # Upload the locally saved model
            fig_blob.upload_from_filename(f"model_acc_{now}.png")
            # Clean up
            os.remove(f"model_acc_{now}.png")
            return jsonify({'message': "Saved the accuracy figure to GCP bucket : " + figure_repo}), 200
        else:
            fig_acc.savefig(f"model_acc_{now}.png")
            return jsonify({'message': 'The Accuracy figure was saved locally.'}), 200
    else:
        return jsonify({'message': 'The history repo not found.'}), 404
