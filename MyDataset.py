# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image
from numpy.fft import fft, ifft, rfft, irfft
from torch.autograd import Variable

def transform(image):
    image = transF.resize(image, size=(300, 600))
    image = transF.to_tensor(image)
    image = transF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

class Data_VIPL(Dataset):
    def __init__(self, root_dir, frames_num, transform = None):
        self.root_dir = root_dir
        self.frames_num = int(frames_num)
        self.datalist = os.listdir(root_dir)
        self.num = len(self.datalist)
        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # print('Inside GetItem function')
        idx = idx
        img_name = 'img_mvavg_full.png'
        # STMap_name = 'STMap_YUV_Align_CSI.png'
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        # print('Path: ',nowPath)
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])                             #path from data should be picked 
        Step_Index = int(temp['Step_Index'])                       # tells about frame number to be picked
        # print("step index: ",Step_Index)
        STMap_Path = os.path.join(nowPath, img_name)

        bvp_name = 'wave.png'
        bvp_path = os.path.join(nowPath, bvp_name)
        # bvp = scio.loadmat(bvp_path)['BVP']
        bvp  = cv2.imread(bvp_path)
        bvp = np.array(bvp.astype('float32')).reshape(-1)
        bvp = bvp[Step_Index:Step_Index + self.frames_num]
        bvp = (bvp - np.min(bvp))/(np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')

        gt_name = 'HR.mat'
        gt_path = os.path.join(nowPath, gt_name)
        gt = scio.loadmat(gt_path)['HR']
        gt = np.array(gt.astype('float32')).reshape(-1)
        gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
        gt = gt.astype('float32')

        # 读取图片序列
        # feature_map = cv2.imread(os.path.join(STMap_Path, STMap_name))
        feature_map = cv2.imread(STMap_Path)
        # print('60 featuremap shape: ',feature_map.shape)
        feature_map = feature_map[:, Step_Index:Step_Index + self.frames_num, :]
        # print('63 featuremap shape: ',feature_map.shape)
        for c in range(feature_map.shape[2]):
            for r in range(feature_map.shape[0]):
                feature_map[r, :, c] = 255 * ((feature_map[r, :, c] - np.min(feature_map[r, :, c])) / (0.00001 +
                            np.max(feature_map[r, :, c]) - np.min(feature_map[r, :, c])))
        feature_map = Image.fromarray(np.uint8(feature_map))
        if self.transform:
            feature_map = self.transform(feature_map)

        # READING FRAME FOR TRANSFORMER MODEL
        video_path = os.path.join(nowPath,"video_crop.avi")
        # print(video_path)

        vidcap  = cv2.VideoCapture(video_path)
        # success, image = vidcap.read()
        # print(success)
        # print(image.shape)

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, Step_Index)
        success,image = vidcap.read()
        image = np.transpose(np.asarray(cv2.resize(image, (64,64))), (2, 0, 1))

        # print("IMAGE SHAPE--->",image.shape)

        # getting video details (fps, amount of frames)
        # fps = vidcap.get(cv2.CAP_PROP_FPS)
        # amountOfFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT )
        # print("fps:",fps)
        # print("Amount of Frames:", amountOfFrames)

        # 归一化
        # print('featuremap shape: ',feature_map.shape)
        return (image,feature_map, bvp, gt, idx)

def CrossValidation(root_dir, fold_num=5,fold_index=0):
    datalist = os.listdir(root_dir)
    # datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    test_num = round(((num/fold_num) - 2))
    train_num = num - test_num
    test_index = datalist[fold_index*test_num:fold_index*test_num + test_num-1]
    train_index = datalist[0:fold_index*test_num] + datalist[fold_index*test_num + test_num:]
    return test_index, train_index

def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    Index_path = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_file in filesList:
        now = os.path.join(root_path, sub_file)
        # img_path = os.path.join(now, os.path.join('STMap', Pic_path))
        img_path = os.path.join(now,Pic_path)
        temp = cv2.imread(img_path)
        Num = temp.shape[1]
        Res = Num - frames_num - 1  # 可能是Diff数据
        Step_num = int(Res/Step)
        for i in range(Step_num):
            Step_Index = i*Step
            temp_path = sub_file + '_' + str(1000 + i) + '_.mat'
            scio.savemat(os.path.join(save_path, temp_path), {'Path': now, 'Step_Index': Step_Index})
            Index_path.append(temp_path)
    return Index_path

