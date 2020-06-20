import os
import warnings
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings('ignore')


def scale_data(data):
    # shape of data: (N, T, C)
    for i in range(data.shape[0]):
        data[i, ...] = preprocessing.scale(data[i, ...])
        # data[i, ...] = preprocessing.minmax_scale(data[i, ...])
        # data[i, ...] = preprocessing.maxabs_scale(data[i, ...])


def load_group_eeg_data(date, group, shuffle=True, sorted_=True):
    if sorted_:
        x = np.load(f'data/2020_{date}_subject_01_sorted_{group:>02}/x_sorted.npy').astype(np.float32)
        y = np.load(f'data/2020_{date}_subject_01_sorted_{group:>02}/y_sorted.npy').astype(np.int64)
    else:
        x = np.load(f'data/2020_{date}_subject_01_sorted_{group:>02}/x_unsorted.npy').astype(np.float32)
        y = np.load(f'data/2020_{date}_subject_01_sorted_{group:>02}/y_unsorted.npy').astype(np.int64)      
    print(f'Original EEG data shape: {x.shape}')
    
    y = y.reshape(-1)
    # transpose and downsample (N, C, T) ===> (N, T/2, C)
    x = np.transpose(x, (0, 2, 1))
    index = np.arange(0, x.shape[1], 2)
    x = x[:, index, :]
    
    # train-test shuffle split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    # preprocess
    scale_data(x_train)
    scale_data(x_test)
    print(f'Processed EEG train data shape: {x_train.shape}')
    print(f'Processed EEG test  data shape: {x_test.shape}')

    return (x_train, x_test, y_train, y_test)


def load_combined_eeg_data(date, sorted_=True):
    if sorted_:
        x_1 = np.load(f'data/2020_{date}_subject_01_sorted_01/x_sorted.npy').astype(np.float32)
        y_1 = np.load(f'data/2020_{date}_subject_01_sorted_01/y_sorted.npy').astype(np.int64)
        x_2 = np.load(f'data/2020_{date}_subject_01_sorted_02/x_sorted.npy').astype(np.float32)
        y_2 = np.load(f'data/2020_{date}_subject_01_sorted_02/y_sorted.npy').astype(np.int64)
    else:
        x_1 = np.load(f'data/2020_{date}_subject_01_unsorted_01/x_unsorted.npy').astype(np.float32)
        y_1 = np.load(f'data/2020_{date}_subject_01_unsorted_01/y_unsorted.npy').astype(np.int64)
        x_2 = np.load(f'data/2020_{date}_subject_01_unsorted_02/x_unsorted.npy').astype(np.float32)
        y_2 = np.load(f'data/2020_{date}_subject_01_unsorted_02/y_unsorted.npy').astype(np.int64)
    
    x, y = np.concatenate((x_1, x_2)), np.concatenate((y_1, y_2))
    print(f'Original EEG data shape: {x.shape}')
    
    y = y.reshape(-1)
    # transpose and downsample (N, C, T) ===> (N, T/2, C)
    x = np.transpose(x, (0, 2, 1))
    index = np.arange(0, x.shape[1], 2)
    x = x[:, index, :]

    # train-test shuffle split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    # preprocess
    scale_data(x_train)
    scale_data(x_test)
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))
    print(f'Processed EEG train data shape: {x_train.shape}')
    print(f'Processed EEG test  data shape: {x_test.shape}')

    return (x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    date = '06_03'
    group = 1
    sorted_ = True
    # x_train, x_test, y_train, y_test = load_group_eeg_data(date, group, sorted_=sorted_)
    x_train, x_test, y_train, y_test = load_combined_eeg_data(date, sorted_=sorted_)
