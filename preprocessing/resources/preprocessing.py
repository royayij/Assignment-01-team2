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
    truth_df = pd.read_csv('PM_truth.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    # MinMax normalization (from 0 to 1)
    test_df['cycle_norm'] = test_df['cycle']
    cols_normalize = test_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize,
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns=test_df.columns)
    test_df = test_df.reset_index(drop=True)

    # We use the ground truth dataset to generate labels for the test data.
    # generate column max for test data
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)
    w1 = 30
    w0 = 15
    # generate RUL for test data
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)

    # generate label columns w0 and w1 for test data
    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
    test_df['label2'] = test_df['label1']
    test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
    return test_df
