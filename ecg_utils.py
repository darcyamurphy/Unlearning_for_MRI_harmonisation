# Darcy Murphy 2021
# Adapted from https://github.com/ZhaoZhibin/AI_Healthcare
import numpy as np
import pandas as pd
import random
from datasets.ecg_dataset import SimpleDataset, LoopingDataset
from config import constants

seq_len = 4096


def load_test_data(test_csv, domain_count, domain_id):
    test_pd = drop_equivalent_classes(pd.read_csv(test_csv))
    test_dataset = build_val_dataset(test_pd, domain_count, domain_id)
    return test_dataset


def load_data(train_csv, val_csv, domain_count, domain_id):
    train_pd = drop_equivalent_classes(pd.read_csv(train_csv))
    val_pd = drop_equivalent_classes(pd.read_csv(val_csv))
    train_dataset = build_train_dataset(train_pd, domain_count, domain_id)
    val_dataset = build_val_dataset(val_pd, domain_count, domain_id)
    return train_dataset, val_dataset


# train and val lists need to be in same order so train and val domains match up
def load_unbalanced_data(train_csvs, val_csvs):
    domain_count = len(train_csvs)
    train_pds, max_train_length = get_pds_from_csvs(train_csvs)
    val_pds, max_val_length = get_pds_from_csvs(val_csvs)
    train_datasets = []
    val_datasets = []
    for i in range(domain_count):
        train_dataset = build_train_dataset(train_pds[i], domain_count, i, max_train_length)
        val_dataset = build_val_dataset(val_pds[i], domain_count, i, max_val_length)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
    return train_datasets, val_datasets


def get_pds_from_csvs(csvs):
    max_length = 0
    pds = []
    for file in csvs:
        train_pd = drop_equivalent_classes(pd.read_csv(file))
        pds.append(train_pd)
        data_length = len(train_pd.index)
        if data_length > max_length:
            max_length = data_length
    return pds, max_length


def build_train_dataset(pd_df, domain_count, domain_id, max_length=None):
    return build_dataset(pd_df, Compose([RandomClip(len=seq_len), Normalize('none'), Retype()]),
                         domain_count, domain_id, max_length)


def build_val_dataset(pd_df, domain_count, domain_id, max_length=None):
    return build_dataset(pd_df, Compose([ValClip(len=seq_len), Normalize('none'), Retype()]),
                         domain_count, domain_id, max_length)


def build_dataset(annotations, transforms, domain_count, domain_id, max_length=None):
    filenames = annotations['filename'].tolist()
    labels = annotations.iloc[:, 4:].values
    age = annotations['age'].tolist()
    gender = annotations['gender'].tolist()
    fs = annotations['fs'].tolist()
    domain = np.zeros((len(filenames), domain_count))
    domain[:,domain_id] = 1
    if max_length is None:
        return SimpleDataset(labels, filenames, age, gender, fs, domain, transform=transforms)
    return LoopingDataset(labels, filenames, age, gender, fs, domain, max_length, transform=transforms)


def drop_equivalent_classes(data):
    # Deal with three equivalent pairs
    droplist = []
    for pair in constants.equivalent_classes:
        data[pair[0]] = data[pair[0]].combine(data[pair[1]], np.maximum)
        droplist.append(pair[1])
    data = data.drop(columns=droplist, axis=1)
    return data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class Normalize(object):
    def __init__(self, type="0-1"):
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            for i in range(seq.shape[0]):
                if np.sum(seq[i, :]) == 0:
                    seq[i, :] = seq[i, :]
                else:
                    seq[i, :] = (seq[i, :] - seq[i, :].min()) / (seq[i, :].max() - seq[i, :].min())
        elif self.type == "mean-std":
            for i in range(seq.shape[0]):
                if np.sum(seq[i, :]) == 0:
                    seq[i, :] = seq[i, :]
                else:
                    seq[i, :] = (seq[i, :] - seq[i, :].mean()) / seq[i, :].std()
        elif self.type == "none":
            seq = seq
        else:
            raise NameError('This normalization is not included!')
        return seq


class RandomClip(object):
    def __init__(self, len=72000):
        self.len = len

    def __call__(self, seq):
        if seq.shape[1] >= self.len:
            start = random.randint(0, seq.shape[1] - self.len)
            seq = seq[:, start:start + self.len]
        else:
            left = random.randint(0, self.len - seq.shape[1])
            right = self.len - seq.shape[1] - left
            zeros_padding1 = np.zeros(shape=(seq.shape[0], left), dtype=np.float32)
            zeros_padding2 = np.zeros(shape=(seq.shape[0], right), dtype=np.float32)
            seq = np.hstack((zeros_padding1, seq, zeros_padding2))
        return seq


class ValClip(object):
    def __init__(self, len=72000):
        self.len = len

    def __call__(self, seq):
        if seq.shape[1] >= self.len:
            start = random.randint(0, seq.shape[1] - self.len)
            seq = seq[:, start:start + self.len]
        else:
            zeros_padding = np.zeros(shape=(seq.shape[0], self.len - seq.shape[1]), dtype=np.float32)
            seq = np.hstack((seq, zeros_padding))
        return seq
