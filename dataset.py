from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data


class MNISTDataset(data.Dataset):
    def __init__(self, filename, train=True):
        data_df = pd.read_csv(filename)
        self.train_flag = train
        if train:
            self.data_label = np.array(data_df['label'].values)
            self.data_label_onehot = self.get_onehot(self.data_label)
            self.data_f_arr = np.array(data_df.iloc[:, 1:])
        else:
            self.data_label = None
            self.data_f_arr = np.array(data_df)
        data_f_arr = self.datapreprocess(self.data_f_arr)
        self.len = len(self.data_f_arr)
        self.data_features = data_f_arr.reshape(self.len,1, 28, 28)

    def __getitem__(self, i):
        index = i % self.len
        if self.train_flag:
            image, label = self.data_features[index], self.data_label_onehot[index]
            return image, label
        else:
            image = self.data_features[index]
            return image

    def __len__(self):
        return len(self.data_features)

    def datapreprocess(self, data):
        ret_data = self.data_normalization(data)
        ret_data = self.data_standardization(ret_data)
        return ret_data

    def labelproprocess(self, label):
        ret_label = self.get_onehot(label)
        return ret_label

    def imageshow(self):
        try:
            plt.figure(figsize=(8, 8))
            for num in range(0, 10):
                image_index = (np.where(self.data_label == num))[0][0]
                plt.subplot(2, 5, num + 1)
                plt.title('number: ' + str(num))
                plt.imshow(self.data_f_arr[image_index].reshape(28, 28))
            plt.show()
        except:
            print('test dataset does not have labels')

    def labelCount(self):
        try:
            counter = Counter(self.data_label)
            plt.bar(list(counter.keys()), counter.values(),
                    color=['slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'indigo',
                           'royalblue', 'lime', 'orange', 'gold', 'cyan'], alpha=0.6)
            for a, b in zip(counter.keys(), counter.values()):
                plt.text(a, b + 1, b, ha='center', va='bottom')
            plt.show()
        except:
            print('test dataset does not have labels')

    # 归一化
    def data_normalization(self, x):
        eps = 1e-9
        x_max = np.max(x, axis=0)
        x_min = np.min(x, axis=0)
        x_new = (x - x_min) / (x_max - x_min + 1e-9)
        return x_new

    # 标准化
    def data_standardization(self, x):
        eps = 1e-9
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x_new = (x - x_mean) / (x_std + eps)
        return x_new

    # label -> onehot
    def get_onehot(self, Y):
        N = len(Y)
        D = len(set(Y))
        label_onehot = np.zeros([N, D])
        for i in range(N):
            id = int(Y[i])
            label_onehot[i, id] = 1
        return label_onehot

