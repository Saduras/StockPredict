import pandas_datareader.data as web
import pandas as pd
import os.path
import numpy as np

import fix_yahoo_finance as yf
yf.pdr_override()

def download_stocks_from_yahoo(symbol, start_date, end_date):
    path = to_path(symbol, start_date, end_date)
    
    df = web.get_data_yahoo(symbol, start = start_date, end = end_date)
    df.to_csv(path)

    return df

def load_stocks(symbol, start_date, end_date):
    path = to_path(symbol, start_date, end_date)

    if os.path.isfile(path):
        return pd.read_csv(path)
    else:
        return download_stocks_from_yahoo(symbol, start_date, end_date)

def pre_pros_as_sequences(dataframe, seq_len, training_fraction = 0.9):
    matrix = dataframe.as_matrix(['Open'])

    # split into sequences
    sequences = []
    for i in range(len(matrix) - seq_len - 1):
        sequences.append(matrix[i : i + seq_len + 1])

    # normalize
    for i in range(len(sequences)):
        sequences[i] = sequences[i] / sequences[i][0] - 1

    sequences = np.array(sequences)

    train_count = int(matrix.shape[0] * training_fraction)
    X_train = sequences[:train_count, :-1]
    Y_train = sequences[:train_count, -1]
    X_test = sequences[train_count:, :-1]
    Y_test = sequences[train_count:, -1]
    return X_train, Y_train, X_test, Y_test

def to_path(symbol, start_date, end_date):
    return './data/' + symbol + '_yahoo_' + str(start_date) + '_to_' + str(end_date) + '.csv'