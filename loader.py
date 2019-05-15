import torch
import cv2
import pandas as pd
import numpy as np
from common.util import alignment, face_ToTensor, get_lbpface
from sklearn import preprocessing
from torch.utils.data import DataLoader


# _lfw_root = '/home/data/Datasets/LFW/lfw/'
_lfw_root = '/home/aaron/Datasets/database/'
_lfw_images = 'data/peopleDevTest.txt'
_lfw_landmarks = 'data/LFW.csv'
_multipie_landmarks = 'data/Multi-PIE.csv'


def load_LFWFace():
    faces = []
    df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
    numpyMatrix = df.values
    landmarks = numpyMatrix[:, 1:]
    with open(_lfw_images) as f:
        images_lines = f.readlines()[1:]
    for index in range(len(images_lines)):
        p = images_lines[index].replace('\n', '').split('\t')
        name = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
        face = alignment(cv2.imread(_lfw_root + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))), landmarks[df.loc[df[0] == name].index.values[0]])
        # lbpface = get_lbpface(face)
        faces.append(face)

    class_id = [path.split('\t')[0] for path in images_lines]
    unique_class_id = np.unique(class_id)
    le = preprocessing.LabelEncoder()
    le.fit(unique_class_id)
    targets = le.transform(class_id).reshape(-1, 1)

    faces = np.asarray(faces, 'uint8')
    # labels = np.asarray(labels, 'uint8')

    return faces, targets


class LFWDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(LFWDataset, self).__init__()
        df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        with open(_lfw_images) as f:
            images_lines = f.readlines()[1:]
        self.images_lines = images_lines
        self.targets, self.num_class = self.get_targets()

    def __getitem__(self, index):
        p = self.images_lines[index].replace('\n', '').split('\t')
        name = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
        face = alignment(cv2.imread(_lfw_root + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))), self.landmarks[self.df.loc[self.df[0] == name].index.values[0]])
        return face_ToTensor(face), torch.LongTensor(self.targets[index])

    def get_targets(self):
        class_id = [path.split('\t')[0] for path in self.images_lines]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.int32), num_class

    def __len__(self):
        return self.targets.shape[0]


class get_loader():
    def __init__(self, batch_size):
        dataset = LFWDataset()
        num_class = None
        num_face = None
        shuffle = False
        drop_last = False
        self.dataloader = DataLoader(dataset=dataset, num_workers=4, batch_size=batch_size, pin_memory=True, shuffle=shuffle, drop_last=drop_last)
        self.num_class = num_class
        self.num_face = num_face
        self.train_iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self.train_iter)
        except:
            self.train_iter = iter(self.dataloader)
            data = next(self.train_iter)
        return data