import pandas as pd
import numpy as np
from sklearn import preprocessing


def pre_process_train_db(dataset):
    rul = pd.DataFrame(dataset.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    dataset = dataset.merge(rul, on=['id'], how='left')
    dataset['RUL'] = dataset['max'] - dataset['cycle']
    dataset.drop('max', axis=1, inplace=True)
    # generate label columns for training data
    # we will only make use of "label1" for binary classification, 
    # while trying to answer the question: is a specific engine going to fail within w1 cycles?
    w1 = 30
    w0 = 15
    dataset['label1'] = np.where(dataset['RUL'] <= w1, 1, 0)
    dataset['label2'] = dataset['label1']
    dataset.loc[dataset['RUL'] <= w0, 'label2'] = 2

    # MinMax normalization (from 0 to 1)
    dataset['cycle_norm'] = dataset['cycle']
    cols_normalize = dataset.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_dataset = pd.DataFrame(min_max_scaler.fit_transform(dataset[cols_normalize]),
                                columns=cols_normalize,
                                index=dataset.index)
    join_df = dataset[dataset.columns.difference(cols_normalize)].join(norm_dataset)
    dataset = join_df.reindex(columns=dataset.columns)

    return dataset


def pre_process_test_db(test_df):
    w1 = 30
    w0 = 15
    # generate RUL for test data
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)

    # generate label columns w0 and w1 for test data
    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
    test_df['label2'] = test_df['label1']
    test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
    return test_df
