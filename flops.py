# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as io
from sklearn.preprocessing import LabelBinarizer
import torch
import MyDataset
import model
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import utils
from timesformer import TimeSformer
import logging
import gc
from linearmodel import MLP
from combined import *
from pytorch_model_summary import summary

import model
from ptflops import get_model_complexity_info

DIM = 128
IMAGE_SIZE = 64
PATCH_SIZE = 8
NUM_FRAMES = 20
DEPTH = 12
HEADS = 8
DIM_HEAD = 64
ATTN_DROPOUT = 0.1
FF_DROPOUT = 0.1
ITERATIONS = 20
tf_learning_rate = 0.001

# DEFINING BVP MODEL PARAMETERS
N_CHANNELS = 3
N_CLASSES = 1

tf_bvp_model = TimeSformer_BVP(dim=DIM, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1, num_frames=NUM_FRAMES, depth=DEPTH,
                                heads=HEADS, dim_head=DIM_HEAD, attn_dropout=ATTN_DROPOUT, ff_dropout=FF_DROPOUT, n_channels=N_CHANNELS, n_classes=N_CLASSES)

tf_model = TimeSformer(dim=DIM, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1, num_frames=NUM_FRAMES, depth=DEPTH,
                       heads=HEADS, dim_head=DIM_HEAD, attn_dropout=ATTN_DROPOUT, ff_dropout=FF_DROPOUT)

unet_model = model.UNet(3, 1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("TF MODEL Parameters: ", count_parameters(tf_model))
print("UNet MODEL Parameters: ", count_parameters(unet_model))
print("TF_BVP MODEL Parameters: ", count_parameters(tf_bvp_model))

unet_flops, params = get_model_complexity_info(unet_model, (3,64,256), as_strings=True, print_per_layer_stat = True)
tf_flops, params = get_model_complexity_info(tf_model, (1,3,64,64), as_strings=True, print_per_layer_stat = True)
print("unet FLOPS: ",unet_flops)
print("tf FLOPS: ",tf_flops)


def prepare_input(resolution):
    video = torch.FloatTensor(1, 1, 3, 64, 64)
    x_Unet = torch.FloatTensor(1, 3, 64, 256)
    return dict(video=video, x_Unet=x_Unet)

flops, params = get_model_complexity_info(tf_bvp_model,((1,3,64,64),(3,64,256)), input_constructor=prepare_input, as_strings=True, print_per_layer_stat = True)
print("tf_bvp FLOPS: ",flops)
