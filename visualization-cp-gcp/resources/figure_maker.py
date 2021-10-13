# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import matplotlib.pyplot as plt
import time
from google.cloud import storage
import os
from flask import jsonify

def make_fig(y):
    fig_verify = plt.figure(figsize=(100, 50))
    plt.plot(y[0], color="blue")
    plt.plot(y[1], color="green")
    plt.title('prediction')
    plt.ylabel('value')
    plt.xlabel('row')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    now = time.time()
    figure_repo = os.environ['FIGURE_REPO']
    if figure_repo:
        # Save the model localy
        fig_verify.savefig(f"model_verify_{now}.png")
        # Save to GCS as model.h5
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(figure_repo)
        blob = bucket.blob(f"model_verify_{now}.png")
        # Upload the locally saved model
        blob.upload_from_filename(f"model_verify_{now}.png")
        # Clean up
        os.remove(f"model_verify_{now}.png")
        return jsonify({'message':"Saved the figure to GCP bucket : " + model_repo}), 200
    else:
        fig_verify.savefig(f"model_verify_{now}.png")
        return jsonify({'message': 'The model was saved locally.'}), 200
