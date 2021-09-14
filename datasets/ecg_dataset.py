from torch.utils.data import Dataset
import torch
import numpy as np
from scipy.io import loadmat


def prepare_data(age, gender):
    data = np.zeros(5, )
    if age >= 0:
        data[0] = age / 100
    if 'F' in gender:
        data[2] = 1
        data[4] = 1
    elif gender == 'Unknown':
        data[4] = 0
    elif 'f' in gender:
        data[2] = 1
        data[4] = 1
    else:
        data[3] = 1
        data[4] = 1

    return data


def load_data(case, src_fs, tar_fs=257):
    x = loadmat(case)
    data = np.asarray(x['val'], dtype=np.float64)
    data = resample(data, src_fs, tar_fs)
    return data


def resample(input_signal, src_fs, tar_fs):
    '''
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''
    if src_fs != tar_fs:
        dtype = input_signal.dtype
        audio_len = input_signal.shape[1]
        audio_time_max = 1.0 * (audio_len) / src_fs
        src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
        tar_time = 1.0 * np.linspace(0, np.int(audio_time_max * tar_fs), np.int(audio_time_max * tar_fs)) / tar_fs
        for i in range(input_signal.shape[0]):
            if i == 0:
                output_signal = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
                output_signal = output_signal.reshape(1, len(output_signal))
            else:
                tmp = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
                tmp = tmp.reshape(1, len(tmp))
                output_signal = np.vstack((output_signal, tmp))
    else:
        output_signal = input_signal
    return output_signal


# implementation of pytorch map-style dataset
class SimpleDataset(Dataset):

    def __init__(self, labels, data, age, gender, fs, domain_id, transform=None, loader=load_data):
        self.data = data
        self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
        self.age = age
        self.gender = gender
        self.fs = fs
        self.transforms = transform
        self.loader = loader
        self.domain = np.ones(len(data)) * domain_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_name = self.data[item]
        fs = self.fs[item]
        age = self.age[item]
        gender = self.gender[item]
        age_gender = prepare_data(age, gender)
        img = self.loader(img_name, src_fs=fs)
        label = self.multi_labels[item]
        img = self.transforms(img)
        domain = self.domain[item]
        return img, torch.from_numpy(label).float(), torch.from_numpy(domain).float()
