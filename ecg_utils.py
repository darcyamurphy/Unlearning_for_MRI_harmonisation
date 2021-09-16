import numpy as np
import pandas as pd
import random
from datasets.ecg_dataset import SimpleDataset


seq_len = 4096


def load_data(train_csv, val_csv, domain_count, domain_id):
    train_pd = drop_equivalent_classes(pd.read_csv(train_csv))
    val_pd = drop_equivalent_classes(pd.read_csv(val_csv))
    train_dataset = build_dataset(train_pd, Compose([RandomClip(len=seq_len), Normalize('none'), Retype()]), domain_count, domain_id)
    val_dataset = build_dataset(val_pd, Compose([ValClip(len=seq_len), Normalize('none'), Retype()]), domain_count, domain_id)
    return train_dataset, val_dataset


def build_dataset(annotations, transforms, domain_count, domain_id):
    filenames = annotations['filename'].tolist()
    labels = annotations.iloc[:, 4:].values
    age = annotations['age'].tolist()
    gender = annotations['gender'].tolist()
    fs = annotations['fs'].tolist()
    domain = np.zeros((len(filenames), domain_count))
    domain[:,domain_id] = 1
    dataset = SimpleDataset(labels, filenames, age, gender, fs, domain, transform=transforms)
    return dataset


def drop_equivalent_classes(data):
    # Deal with three equivalent pairs
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    droplist = []
    for pair in equivalent_classes:
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
