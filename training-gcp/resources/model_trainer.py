# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import logging

from flask import jsonify
from google.cloud import storage
import keras
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

# pick a large window size of 50 cycles
sequence_length = 50


# function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 111 191 -> from row 111 to 191
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


# function to generate labels
def gen_labels(id_df, seq_length, label):
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]


def train(train_df):
    model_path = '../../Output/binary_model.h5'
    # pick the feature columns
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)
    # generator for the sequences
    seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], sequence_length, sequence_cols))
               for id in train_df['id'].unique())

    # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    seq_array.shape
    # generate labels
    label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['label1'])
                 for id in train_df['id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)
    label_array.shape
    # Next, we build a deep network.
    # The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units.
    # Dropout is also applied after each LSTM layer to control overfitting.
    # Final layer is a Dense output layer with single unit and sigmoid activation since this is a binary classification problem.
    # build the network
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]

    model = Sequential()

    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=nb_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
              callbacks=[
                  keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
                                                mode='min'),
                  keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True,
                                                  mode='min', verbose=0)]
              )
    scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
    text_out = {
        "accuracy:": scores[1],
        "loss": scores[0],
    }
    project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
    model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
    if model_repo:
        # Save the model localy
        model.save('local_model.h5')
        # Save to GCS as model.h5
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(model_repo)
        blob = bucket.blob('model.h5')
        # Upload the locally saved model
        blob.upload_from_filename('local_model.h5')
        # Clean up
        os.remove('local_model.h5')
        logging.info("Saved the model to GCP bucket : " + model_repo)
        return jsonify(text_out), 200
    else:
        model.save("model.h5")
        return jsonify({'message': 'The model was saved locally.'}), 200
