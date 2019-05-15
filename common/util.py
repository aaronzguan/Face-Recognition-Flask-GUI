import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np
import cv2
from .matlab_cp2tform import get_similarity_transform_for_cv2
import pandas as pd
import os
import sys
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern

def get_lbpface(face):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), n_points, radius)
    lbp = np.uint8(lbp)
    lbp = lbp.reshape(-1)
    lbp_desc, _ = np.histogram(lbp, bins=256, range=(0, 255))
    # hist = []
    # hist = np.asarray(hist).swapaxes(1, 0)
    # hist = hist.reshape(-1, 7, 6)
    return lbp_desc


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i*n/n_folds):int((i+1)*n/n_folds)]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds


def eval_acc(threshold, diff):
    y_predict = np.int32(diff[:,0]>threshold)
    y_true = np.int32(diff[:,1])
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def save_results(results, path):
    data_frame = pd.DataFrame(data=results)
    data_frame.to_csv(path + 'train_results.csv')


def alignment(src_img, src_pts, size=None):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655],
               [62.7299, 92.2041]]
    if size is not None:
        ref_pts = np.array(ref_pts)
        ref_pts[:,0] = ref_pts[:,0] * size/96
        ref_pts[:,1] = ref_pts[:,1] * size/96
        crop_size = (int(size), int(112/(96/size)))
    else:
        crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5, 2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size, flags=cv2.INTER_CUBIC)
    if size is not None:
        face_img = cv2.resize(face_img, dsize=(96, 112), interpolation=cv2.INTER_CUBIC)
    return face_img


def crop_align_resize(src_img, src_pts):
    src_pts = np.array(src_pts).reshape(5, 2)
    y_min, y_max = src_pts[:, 1].min(), src_pts[:, 1].max()
    size = int((y_max-y_min) * 3)
    crop_img = alignment(src_img, src_pts, size)
    return crop_img


class L2Norm(nn.Module):
    def forward(self, input, dim=1):
        return F.normalize(input, p=2, dim=dim)


def face_ToTensor(img):
    return (ToTensor()(img) - 0.5) * 2


def save_log(losses, args):
    message = ''
    for name, value in losses.items():
        message += '{}: {:.4f} '.format(name, value)
    log_name = os.path.join(args.checkpoints_dir, args.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('\n' + message)


def tensor_pair_cosine_distance(features11, features12, features21, features22, type='normal'):
    if type == 'concat':
        features1 = torch.cat((features11, features12), dim=1)
        features2 = torch.cat((features21, features22), dim=1)
    elif type == 'sum':
        features1 = features11 + features12
        features2 = features21 + features22
    elif type == 'normal':
        features1 = features11
        features2 = features21
    else:
        print('tensor_pair_cosine_distance unspported type!')
        sys.exit()
    scores = torch.nn.CosineSimilarity()(features1, features2)
    scores = scores.cpu().numpy().reshape(-1, 1)
    return scores


def tensor_pair_cosine_distance_matrix(features11, features12, features21, features22, type='normal'):
    if type == 'concat':
        features1 = torch.cat((features11, features12), dim=1)
        features2 = torch.cat((features21, features22), dim=1)
    elif type == 'sum':
        features1 = features11 + features12
        features2 = features21 + features22
    elif type == 'normal':
        features1 = features11
        features2 = features21
    else:
        print('tensor_pair_cosine_distance_matrix unspported type!')
        sys.exit()
    features1_np = features1.cpu().numpy()
    features2_np = features2.cpu().numpy()
    scores = 1 - cdist(features2_np, features1_np, 'cosine')
    return scores