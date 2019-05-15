import torch
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn import preprocessing
import scipy.io as sio
from .util import alignment, crop_align_resize, face_ToTensor, L2Norm
from PIL import Image
from skimage.feature import local_binary_pattern

# _casia_root = '/home/data/Datasets/CASIA-WebFace/CASIA-WebFace/'
# _casia_landmarks = '../data/CASIA-maxpy-clean_remove_lfw_megaface.csv'
# _vggface2_root = '/home/data/Datasets/VGGFace2/train_test_112x96/'
# _vggface2_landmarks ='/home/data/Datasets/VGGFace2/bb_landmark/loose_landmark_train_test_remove_lfw_megaface.csv'
# _lfw_root = '/home/data/Datasets/LFW/lfw/'
# _lfw_landmarks = '../data/LFW.csv'
# _lfw_pairs = '../data/lfw_pairs.txt'
# _scface_root = '../../../Datasets/SCface/SCface_database/'
# _scface_mugshot_landmarks = '../data/scface_mugshot.csv'
# _scface_camera_landmarks = '../data/{}.csv'
# _survface_root= '/home/data/Datasets/QMUL-SurvFace/QMUL-SurvFace/'


_casia_root = '/home/johnny/Datasets/CASIA-WebFace/CASIA-WebFace/'
_casia_landmarks = '../data/CASIA-maxpy-clean_remove_lfw.csv'
_vggface2_root = '/home/johnny/Datasets/VGGFace2/train_test_112x96/'
_vggface2_landmarks ='/home/johnny/Datasets/VGGFace2/bb_landmark/loose_landmark_train_test_remove_lfw_megaface.csv'
_lfw_root = '/home/johnny/Datasets/LFW/lfw/'
_lfw_landmarks = '../data/LFW.csv'
_lfw_pairs = '../data/lfw_pairs.txt'
_scface_root = '../../../Datasets/SCface/SCface_database/'
_scface_mugshot_landmarks = '../data/scface_mugshot.csv'
_scface_camera_landmarks = '../data/{}.csv'
_survface_root= '/home/johnny/Datasets/QMUL-SurvFace/QMUL-SurvFace/'

_tinyface_root = '/home/johnny/Datasets/QMUL-TinyFace/tinyface/'
_tinyface_probe = '../data/tinyface_probe_img_ID_pairs.csv'
_tinyface_gallery_match = '../data/tinyface_gallery_img_ID_pairs.csv'

_multipie_root = '/home/Datasets/Multi-PIE/session01/multiview/'
_multipie_landmarks = '../data/Multi-PIE.csv'

_blufr_test_set = '../data/blufr_test_set.txt'
_lfw_labels = '../data/lfw_labels.txt'
# 10 blufr trials
blufr_trials = [10281, 9805, 9388, 9447, 9470, 9858, 9952, 9511, 9452, 9919]


class TinyFaceTrainDataset(torch.utils.data.Dataset):
    def __init__(self, is_aug=False):
        super(TinyFaceTrainDataset, self).__init__()
        self.faces_path = glob(_tinyface_root + 'Training_Set/*/*')
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug

    def __getitem__(self, index):
        face = cv2.imread(self.faces_path[index])
        face = cv2.resize(face, (96, 112), cv2.INTER_CUBIC)  ## ???? resize or not??

        if self.is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)

        return face_ToTensor(face), torch.LongTensor(self.targets[index])

    def __len__(self):
        return len(self.faces_path)

    def get_targets(self):
        class_id = [path.split(_tinyface_root + 'Training_Set/')[1].split('/')[0] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.int32), num_class


class TinyFaceProbeDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TinyFaceProbeDataset, self).__init__()
        self.probe_faces_path = _tinyface_root + 'Testing_Set/Probe/'
        df = pd.read_csv(_tinyface_probe, delimiter=",", header=None)
        numpyMatrix = df.values
        self.probe_img_name = numpyMatrix[:, 0]
        self.probe_label = numpyMatrix[:, 1]

    def __getitem__(self, index):
        probe_face = cv2.imread(self.probe_faces_path + self.probe_img_name[index])
        probe_face = cv2.resize(probe_face, (96, 112), cv2.INTER_CUBIC)
        probe_face_flip = cv2.flip(probe_face, 1)

            # print(probe_face.shape)
            # ?? need normalization ??
        return face_ToTensor(probe_face), face_ToTensor(probe_face_flip), torch.LongTensor(np.int32(self.probe_label[index]).reshape(1))

    def __len__(self):
        return len(self.probe_img_name)


class TinyFaceGalleryDataset(torch.utils.data.Dataset):
    def __init__(self, is_gallery_match=True):
        super(TinyFaceGalleryDataset, self).__init__()

        if is_gallery_match:
            self.gallery_match_faces_path = _tinyface_root + 'Testing_Set/Gallery_Match/'
            df = pd.read_csv(_tinyface_gallery_match, delimiter=",", header=None)
            numpyMatrix = df.values
            self.gallery_match_img_name = numpyMatrix[:, 0]
            self.gallery_match_label = numpyMatrix[:, 1]
        else:
            self.gallery_distract_faces_path = glob(_tinyface_root + 'Testing_Set/Gallery_Distractor/*')
            self.gallery_distract_label = np.int32(-100).reshape(1)

        self.is_gallery_match = is_gallery_match

    def __getitem__(self, index):
        if self.is_gallery_match:
            gallery_match_face = cv2.imread(self.gallery_match_faces_path + self.gallery_match_img_name[index])
            gallery_match_face = cv2.resize(gallery_match_face, (96, 112), cv2.INTER_CUBIC)
            gallery_match_face_flip = cv2.flip(gallery_match_face, 1)

            return face_ToTensor(gallery_match_face), face_ToTensor(gallery_match_face_flip), torch.LongTensor(np.int32(self.gallery_match_label[index]).reshape(1))
        else:
            gallery_distract_face = cv2.imread(self.gallery_distract_faces_path[index])
            gallery_distract_face = cv2.resize(gallery_distract_face, (96, 112), cv2.INTER_CUBIC)
            gallery_distract_face_flip = cv2.flip(gallery_distract_face, 1)

            return face_ToTensor(gallery_distract_face), face_ToTensor(gallery_distract_face_flip), torch.LongTensor(self.gallery_distract_label)

    def __len__(self):
        if self.is_gallery_match:
            return len(self.gallery_match_img_name)
        else:
            return len(self.gallery_distract_faces_path)


class MultiPIEDataset(torch.utils.data.Dataset):
    def __init__(self, args, pose_angle='15', light_degree=None, select_gallery=True):
        super(MultiPIEDataset, self).__init__()
        df = pd.read_csv(_multipie_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:, 0]
        self.landmarks = numpyMatrix[:, 1:]
        self.targets, self.num_class = self.get_targets()
        self.select_gallery = select_gallery
        self.pose_angle = pose_angle
        self.light_degree = light_degree
        if self.select_gallery:
            self.path, self.landmarks, self.labels = self.get_gallery()
        else:
            self.path, self.landmarks, self.labels = self.get_probe()
        self.num_face = len(self.path)
        self.args = args

    def __getitem__(self, index):
        try:
            face = cv2.imread(_multipie_root + self.path[index])
            face = alignment(face, self.landmarks[index].reshape(-1, 2))
        except:
            face = Image.open(_multipie_root + self.path[index])
            face = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
            face = alignment(face, self.landmarks[index].reshape(-1, 2))
        face_flip = cv2.flip(face, 1)

        if self.args.use_lbp:
            face = torch.cat((face_ToTensor(face), get_lbpface_tensor(face)), 0)
            face_flip = torch.cat((face_ToTensor(face_flip), get_lbpface_tensor(face_flip)), 0)
            return face, face_flip, torch.LongTensor(self.labels[index])
            # return get_lbpface_tensor(face), get_lbpface_tensor(face_flip),  torch.LongTensor(self.labels[index])

        else:
            return face_ToTensor(face), face_ToTensor(face_flip), torch.LongTensor(self.labels[index])

        # return face_ToTensor(face), face_ToTensor(face_flip), torch.LongTensor(self.labels[index])

    def __len__(self):
        return self.num_face

    def get_targets(self):
        class_id = [path.split('/')[0] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1,1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.float32), num_class

    def get_gallery(self):
        gallery_path = []
        gallery_landmarks = []
        gallery_label = []
        for index in range(self.faces_path.shape[0]):
            # Use record number 01
            if self.faces_path[index].split('/')[1] == '01':
                if self.faces_path[index].split('/')[-1].split('_')[-2] == '051' and \
                        self.faces_path[index].split('/')[-1].split('_')[-1] == '07.png':
                    gallery_path += [self.faces_path[index]]
                    gallery_landmarks += [self.landmarks[index]]
                    gallery_label += [self.targets[index]]
        gallery_landmarks = np.asarray(gallery_landmarks)
        return gallery_path, gallery_landmarks, gallery_label

    def get_probe(self):
        probe_path = []
        probe_landmarks = []
        probe_label = []
        for index in range(self.faces_path.shape[0]):
            # Use record number 01
            if self.faces_path[index].split('/')[1] == '01':
                if self.pose_angle is None and self.light_degree is not None:
                    # Skip the gallery image
                    if self.faces_path[index].split('/')[-1].split('_')[-2] == '051' and self.light_degree == '07':
                        pass
                    else:
                        light_suffix = self.light_degree + '.png'
                        if self.faces_path[index].split('/')[-1].split('_')[-1] == light_suffix:
                            probe_path += [self.faces_path[index]]
                            probe_landmarks += [self.landmarks[index]]
                            probe_label += [self.targets[index]]
                elif self.pose_angle is not None:
                    if self.pose_angle == '15':
                        cam1 = '05_0'
                        cam2 = '05_0'
                    elif self.pose_angle == '30':
                        cam1 = '04_1'
                        cam2 = '04_1'
                    elif self.pose_angle == '45':
                        cam1 = '19_0'
                        cam2 = '19_0'
                    elif self.pose_angle == '60':
                        cam1 = '20_0'
                        cam2 = '20_0'
                    elif self.pose_angle == '75':
                        cam1 = '01_0'
                        cam2 = '01_0'
                    elif self.pose_angle == '90':
                        cam1 = '24_0'
                        cam2 = '24_0'
                    elif self.pose_angle == '-15':
                        cam1 = '14_0'
                        cam2 = '14_0'
                    elif self.pose_angle == '-30':
                        cam1 = '13_0'
                        cam2 = '13_0'
                    elif self.pose_angle == '-45':
                        cam1 = '08_0'
                        cam2 = '08_0'
                    elif self.pose_angle == '-60':
                        cam1 = '09_0'
                        cam2 = '09_0'
                    elif self.pose_angle == '-75':
                        cam1 = '12_0'
                        cam2 = '12_0'
                    elif self.pose_angle == '-90':
                        cam1 = '11_0'
                        cam2 = '11_0'
                    elif self.pose_angle == '+-15':
                        cam1 = '05_0'
                        cam2 = '14_0'
                    elif self.pose_angle == '+-30':
                        cam1 = '04_1'
                        cam2 = '13_0'
                    elif self.pose_angle == '+-45':
                        cam1 = '19_0'
                        cam2 = '08_0'
                    elif self.pose_angle == '+-60':
                        cam1 = '20_0'
                        cam2 = '09_0'
                    elif self.pose_angle == '+-75':
                        cam1 = '01_0'
                        cam2 = '12_0'
                    elif self.pose_angle == '+-90':
                        cam1 = '24_0'
                        cam2 = '11_0'
                    else:
                        raise Exception('pose_angle is not correct')
                    if self.light_degree is None:
                        if self.faces_path[index].split('/')[2] == cam1 or self.faces_path[index].split('/')[2] == cam2:
                            probe_path += [self.faces_path[index]]
                            probe_landmarks += [self.landmarks[index]]
                            probe_label += [self.targets[index]]
                    else:
                        light_suffix = self.light_degree + '.png'
                        if self.faces_path[index].split('/')[-1].split('_')[-1] == light_suffix and \
                                (self.faces_path[index].split('/')[2] == cam1 or self.faces_path[index].split('/')[2] == cam2):
                            probe_path += [self.faces_path[index]]
                            probe_landmarks += [self.landmarks[index]]
                            probe_label += [self.targets[index]]
        probe_landmarks = np.asarray(probe_landmarks)
        return probe_path, probe_landmarks, probe_label


class WebFaceDataset(torch.utils.data.Dataset):
    def __init__(self, args, size=False, is_aug=True):
        super(WebFaceDataset, self).__init__()
        # Original num. subjects(num. images): 10,575(494,414)
        df = pd.read_csv(_casia_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:, 0]
        self.landmarks = numpyMatrix[:, 1:]
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug
        self.size = size
        self.args = args

    def __getitem__(self, index):
        face = cv2.imread(_casia_root + self.faces_path[index])
        face = alignment(face, self.landmarks[index].reshape(-1,2))
        if self.is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)

        if self.args.use_lbp:
            # face = torch.cat((face_ToTensor(face), get_lbpface_tensor(face)), 0)
            # return face, torch.LongTensor(self.targets[index])
            return get_lbpface_tensor(face), torch.LongTensor(self.targets[index])
        else:
            return face_ToTensor(face), torch.LongTensor(self.targets[index])

    def __len__(self):
        return self.faces_path.shape[0]

    def get_targets(self):
        class_id = [path.split('/')[0] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1,1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.float32), num_class


class VGGFace2Dataset(torch.utils.data.Dataset):
    def __init__(self, args, is_aug=True):
        super(VGGFace2Dataset, self).__init__()
        # Original #subjects(#images): xxx(3141890)
        df = pd.read_csv(_vggface2_landmarks, delimiter=",")
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:, 0]
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug
        self.args = args

    def __getitem__(self, index):
        face = cv2.imread(_vggface2_root + self.faces_path[index] + '.jpg')
        if self.is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)

        if self.args.use_lbp:
            face = torch.cat((face_ToTensor(face), get_lbpface_tensor(face)), 0)
            return face, torch.LongTensor(self.targets[index])
        else:
            return face_ToTensor(face), torch.LongTensor(self.targets[index])

    def __len__(self):
        return self.faces_path.shape[0]

    def get_targets(self):
        class_id = [path.split('/')[1] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.int32), num_class


class LFWDataset(torch.utils.data.Dataset):
    def __init__(self, args, img_size=None):
        super(LFWDataset, self).__init__()
        df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        with open(_lfw_pairs) as f:
            pairs_lines = f.readlines()[1:]
        self.pairs_lines = pairs_lines
        self.img_size = img_size
        self.args = args

    def __getitem__(self, index):
        p = self.pairs_lines[index].replace('\n', '').split('\t')
        if 3 == len(p):
            sameflag = np.int32(1).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = np.int32(0).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = alignment(cv2.imread(_lfw_root + name1),
                         self.landmarks[self.df.loc[self.df[0] == name1].index.values[0]])
        img2 = alignment(cv2.imread(_lfw_root + name2),
                         self.landmarks[self.df.loc[self.df[0] == name2].index.values[0]])
        if self.img_size is not None:
            """
                Downsample and upsample the second image
            """
            img2 = cv2.resize(img2, (self.img_size, self.img_size), cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, (96, 112), cv2.INTER_CUBIC)
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)

        if self.args.use_lbp:
            # img1 = torch.cat((face_ToTensor(img1), get_lbpface_tensor(img1)), 0)
            # img2 = torch.cat((face_ToTensor(img2), get_lbpface_tensor(img2)), 0)
            # img1_flip = torch.cat((face_ToTensor(img1_flip), get_lbpface_tensor(img1_flip)), 0)
            # img2_flip = torch.cat((face_ToTensor(img2_flip), get_lbpface_tensor(img2_flip)), 0)
            # return img1, img2, \
            #    img1_flip, img2_flip, \
            #    torch.LongTensor(sameflag)
            return get_lbpface_tensor(img1), get_lbpface_tensor(img2), \
                   get_lbpface_tensor(img1_flip), get_lbpface_tensor(img2_flip), \
                   torch.LongTensor(sameflag)
        else:
            return face_ToTensor(img1), face_ToTensor(img2), \
               face_ToTensor(img1_flip), face_ToTensor(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pairs_lines)


class BLUFRDataset(torch.utils.data.Dataset):
    def __init__(self, trialIndex):
        super(BLUFRDataset, self).__init__()
        df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        lineStart = 2
        if trialIndex == 0:
            lineStart = 2
        else:
            for i in range(trialIndex):
                lineStart += (blufr_trials[i] + 1)
        lineEnd = lineStart + blufr_trials[trialIndex] - 1
        with open(_blufr_test_set) as f:
            lines = f.readlines()[0:]
        self.lines = lines[lineStart:lineEnd]
        with open(_lfw_labels) as f:
            labels = f.readlines()[0:]
        self.labels = labels

        assert (0 <= trialIndex <= 9), print('trialIndex should be in the range of [0,9]')

    def __getitem__(self, index):
        p = self.lines[index].replace('\n', '').split(', ')
        name = p[1][:-9] + '/' + p[1]
        img = alignment(cv2.imread(_lfw_root + name), self.landmarks[self.df.loc[self.df[0] == name].index.values[0]])
        img_flip = cv2.flip(img, 1)
        targets = np.int32(self.labels[int(p[0]) - 1]).reshape(1)
        return face_ToTensor(img), face_ToTensor(img_flip), torch.LongTensor(targets)

    def __len__(self):
        return len(self.lines)


class SurvFaceDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_aug):
        super(SurvFaceDataset, self).__init__()
        # mean_h, mean_w : 24.325, 19.92
        self.faces_path = glob(_survface_root + 'training_set/*/*')
        self.targets, self.num_class = self.get_targets()
        self.is_aug = is_aug
        self.args = args

    def __getitem__(self, index):
        face = cv2.imread(self.faces_path[index])
        face = cv2.resize(face, (96, 112), cv2.INTER_CUBIC)
        if self.is_aug:
            if np.random.random() > 0.5:
                face = cv2.flip(face, 1)
        if self.args.use_lbp:
            face = torch.cat((face_ToTensor(face), get_lbpface_tensor(face)), 0)
            return face, torch.LongTensor(self.targets[index])
        else:
            return face_ToTensor(face), torch.LongTensor(self.targets[index])

    def __len__(self):
        return len(self.faces_path)

    def get_targets(self):
        class_id = [path.split(_survface_root + 'training_set/')[1].split('/')[0]
                    for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.int32), num_class


class SurvFace_verification(torch.utils.data.Dataset):
    def __init__(self, args):
        super(SurvFace_verification, self).__init__()
        self.root = _survface_root + 'Face_Verification_Test_Set/'
        self.image_root = self.root + 'verification_images/'
        pos = sio.loadmat(self.root + 'positive_pairs_names.mat')['positive_pairs_names']
        neg = sio.loadmat(self.root + 'negative_pairs_names.mat')['negative_pairs_names']
        self.image_files = np.vstack((pos, neg))
        self.labels = np.vstack((np.ones(len(pos)).reshape(-1, 1),
                                 np.zeros(len(neg)).reshape(-1, 1)))
        self.args = args

    def __getitem__(self, index):
        sameflag = self.labels[index]
        img1 = cv2.imread(self.image_root + self.image_files[index][0][0])
        img2 = cv2.imread(self.image_root + self.image_files[index][1][0])
        # Resize
        img1 = cv2.resize(img1, (96, 112), cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, (96, 112), cv2.INTER_CUBIC)
        # Mirror face trick
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)

        if self.args.use_lbp:
            # img1 = torch.cat((face_ToTensor(img1), get_lbpface_tensor(img1)), 0)
            # img2 = torch.cat((face_ToTensor(img2), get_lbpface_tensor(img2)), 0)
            # img1_flip = torch.cat((face_ToTensor(img1_flip), get_lbpface_tensor(img1_flip)), 0)
            # img2_flip = torch.cat((face_ToTensor(img2_flip), get_lbpface_tensor(img2_flip)), 0)
            # return img1, img2, \
            #        img1_flip, img2_flip, \
            #        torch.LongTensor(sameflag)
            return get_lbpface_tensor(img1), get_lbpface_tensor(img2), \
                   get_lbpface_tensor(img1_flip), get_lbpface_tensor(img2_flip), \
                   torch.LongTensor(sameflag)
        else:
            return face_ToTensor(img1), face_ToTensor(img2), \
               face_ToTensor(img1_flip), face_ToTensor(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.labels)


class SCface_mugshot(torch.utils.data.Dataset):
    def __init__(self):
        super(SCface_mugshot, self).__init__()
        df = pd.read_csv(_scface_mugshot_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:, 0]
        assert len(self.faces_path) == 130, "Wrong number of SCface mugshot"
        self.landmarks = numpyMatrix[:, 1:]
        self.targets, _ = self.get_targets()

    def __getitem__(self, index):
        face = cv2.imread(_scface_root + self.faces_path[index])
        face = alignment(face, self.landmarks[index].reshape(-1,2))
        # output = 'output/' + self.faces_path[index].split('/')[0] + '/'
        # if not os.path.exists(output):
        #     os.makedirs(output)
        # cv2.imwrite('output/' + self.faces_path[index], face)
        face_flip = cv2.flip(face, 1)
        return face_ToTensor(face), face_ToTensor(face_flip), torch.LongTensor(self.targets[index])

    def __len__(self):
        return self.faces_path.shape[0]

    def get_targets(self):
        class_id = [path.split('/')[1].split('_')[0] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.int32), num_class

    def __len__(self):
        return len(self.faces_path)


class SCface_camera(torch.utils.data.Dataset):
    def __init__(self, name):
        super(SCface_camera, self).__init__()
        df = pd.read_csv(_scface_camera_landmarks.format(name), delimiter=",", header=None)
        numpyMatrix = df.values
        self.faces_path = numpyMatrix[:, 0]
        assert len(self.faces_path) == 650, "Wrong number of SCface {}".format(name)
        self.landmarks = numpyMatrix[:, 1:]
        self.targets, _= self.get_targets()

    def __getitem__(self, index):
        face = cv2.imread(_scface_root + self.faces_path[index])
        face = crop_align_resize(face, self.landmarks[index].reshape(-1, 2))
        # output = 'output/' + self.faces_path[index].split('/')[0] + '/'
        # if not os.path.exists(output):
        #     os.makedirs(output)
        # cv2.imwrite('output/' + self.faces_path[index], face)
        face_flip = cv2.flip(face, 1)
        return face_ToTensor(face), face_ToTensor(face_flip), torch.LongTensor(self.targets[index])

    def __len__(self):
        return self.faces_path.shape[0]

    def get_targets(self):
        class_id = [path.split('/')[1].split('_')[0] for path in self.faces_path]
        unique_class_id = np.unique(class_id)
        le = preprocessing.LabelEncoder()
        le.fit(unique_class_id)
        targets = le.transform(class_id).reshape(-1, 1)
        num_class = unique_class_id.shape[0]
        return targets.astype(np.int32), num_class

    def __len__(self):
        return len(self.faces_path)


class get_loader():
    def __init__(self, args, name, batch_size, multipie_pose='15', multipie_light=None, is_aug=True, shuffle=True, drop_last=True, img_size=None, trialIndex=None, tinyface_gallery_match=True):
        if name == 'webface':
            dataset = WebFaceDataset(args, is_aug=is_aug)
            num_class = dataset.num_class
            num_face = None
        elif name == 'vggface2':
            dataset = VGGFace2Dataset(args, is_aug=is_aug)
            num_class = dataset.num_class
            num_face = None
        elif name == 'survface':
            dataset = SurvFaceDataset(args, is_aug=is_aug)
            num_class = dataset.num_class
            num_face = None
        elif name == 'lfw':
            dataset = LFWDataset(args, img_size)
            num_class = None
            num_face = None
            shuffle = False
            drop_last = False
        elif name == 'survface_verification':
            dataset = SurvFace_verification(args)
            num_class = None
            num_face = None
            shuffle = False
            drop_last = False
        elif name == 'scface_mugshots':
            dataset = SCface_mugshot()
            num_class = None
            num_face = None
            shuffle = False
            drop_last = False
        elif 'scface_distance' in name:
            dataset = SCface_camera(name)
        elif name == 'blufr':
            dataset = BLUFRDataset(trialIndex=trialIndex)
            num_class = None
            num_face = None
            shuffle = False
            drop_last = False
        elif name =='tinyface':
            dataset = TinyFaceTrainDataset(is_aug=is_aug)
            num_class = dataset.num_class
            num_face = None
        elif name =='tinyface_probe':
            dataset = TinyFaceProbeDataset()
            num_class = None
            num_face = None
            shuffle = False
            drop_last = False
        elif name =='tinyface_gallery':
            dataset = TinyFaceGalleryDataset(is_gallery_match=tinyface_gallery_match)
            num_class = None
            num_face = None
            shuffle = False
            drop_last = False
        elif name == 'multiPIE_gallery':
            dataset = MultiPIEDataset(args, pose_angle=multipie_pose, light_degree=multipie_light, select_gallery=True)
            num_class = None
            num_face = dataset.num_face
            shuffle = False
            drop_last = False
        elif name == 'multiPIE_probe':
            dataset = MultiPIEDataset(args, pose_angle=multipie_pose, light_degree=multipie_light, select_gallery=False)
            num_class = None
            num_face = dataset.num_face
            shuffle = False
            drop_last = False

        self.dataloader = DataLoader(dataset=dataset, num_workers=4, batch_size=batch_size,
                                     pin_memory=True, shuffle=shuffle, drop_last=drop_last)
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


def get_lbpface_tensor(face):

    # ## Method 1
	# radius = 1
	# n_points = 8 * radius
	# lbp = local_binary_pattern(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), n_points, radius)
	# lbp = np.uint8(lbp)
	# n_bins = int(lbp.max() + 1)
	# lbp_tensor = face_ToTensor(np.expand_dims(lbp, axis=2))
	# hist = []
	# lbp_regions = split(lbp, 4, 4)
	# for region in lbp_regions:
	# 	region = region.reshape(-1)
	# 	lbp_desc, _ = np.histogram(region, bins=n_bins, density=True)
	# 	hist += [lbp_desc]
	# hist = np.asarray(hist).swapaxes(1, 0)
	# hist = hist.reshape(-1, 7, 6)
	# return torch.Tensor(hist)

	## Method 2
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), n_points, radius)
    lbp = np.uint8(lbp)
    lbp = np.expand_dims(lbp, 2)

    return face_ToTensor(lbp)

def split(array, nrows, ncols):
	"""Split a matrix into sub-matrices."""
	r, h = array.shape
	return (array.reshape(h // nrows, nrows, -1, ncols)
	        .swapaxes(1, 2)
	        .reshape(-1, nrows, ncols))