import unetPart
import sys
import torch.nn as nn
from torch import nn, einsum
import torch
import cv2
import numpy as np
import os
import math
import torch.optim as optim


import torch.nn.functional as F
from einops import rearrange, repeat
sys.path.append('..')

##################################### TRANSFORMER SECTION DEPENDENCIES #######################################


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # q *= self.scale                #commented only bcoz inplace not working in flops calculation
        tempq = q * self.scale  # using this only for flops calucation
        q = tempq

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(
            t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(
            t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # combine heads out
        return self.to_out(out)


# # class TimeSformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         num_frames,
#         num_classes,
#         image_size=64,
#         patch_size=16,
#         channels=3,
#         depth=12,
#         heads=8,
#         dim_head=64,
#         attn_dropout=0.,
#         ff_dropout=0.
#     ):
#         super().__init__()
#         assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_size // patch_size) ** 2
#         num_positions = num_frames * num_patches
#         patch_dim = channels * patch_size ** 2

#         self.patch_size = patch_size
#         self.to_patch_embedding = nn.Linear(patch_dim, dim)
#         self.pos_emb = nn.Embedding(num_positions + 1, dim)
#         self.cls_token = nn.Parameter(torch.randn(1, dim))

#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, dim_head=dim_head,
#                         heads=heads, dropout=attn_dropout)),  # Time attention
#                 PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads,
#                         dropout=attn_dropout)),  # Spatial attention
#                 # Feed Forward
#                 PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
#             ]))

#         self.to_out = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#             # nn.Linear(dim,1)
#         )

#         self.transformer_features = nn.LayerNorm(dim)

#     # def forward(self, video):
#         b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
#         assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

#         n = (h // p) * (w // p)
#         video = rearrange(
#             video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)

#         tokens = self.to_patch_embedding(video)

#         cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
#         x = torch.cat((cls_token, tokens), dim=1)
#         x += self.pos_emb(torch.arange(x.shape[1], device=device))

#         for (time_attn, spatial_attn, ff) in self.layers:
#             x = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
#             x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f) + x
#             x = ff(x) + x

#         cls_token = x[:, 0]

#         return self.to_out(cls_token), self.transformer_features(cls_token)


##################################### BVPnet SECTION DEPENDENCIES #######################################


# -*- coding: UTF-8 -*-



# # class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = unetPart.DoubleConv(n_channels, 48)
#         self.down1 = unetPart.Down(48, 96)
#         self.down2 = unetPart.Down(96, 192)
#         self.down3 = unetPart.Down(192, 384)
#         self.down4 = unetPart.Down(384, 384)
#         self.up1 = unetPart.Up(768, 192, bilinear)
#         self.up2 = unetPart.Up(384, 96, bilinear)
#         self.up3 = unetPart.Up(192, 48, bilinear)
#         self.up4 = unetPart.Up(96, 48, bilinear)
#         self.outc = unetPart.OutConv(48, n_classes)
#         self.HR = nn.Sequential(BasicBlock(384, 256, 2, downsample=1, Res=1),
#                                 BasicBlock(256, 256, 1, downsample=0, Res=0),
#                                 BasicBlock(256, 512, 2, downsample=1, Res=1),
#                                 BasicBlock(512, 512, 1, downsample=0, Res=0)
#                                 )
#         self.av = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, 1)

#     # def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         HR = self.fc(torch.squeeze(self.av(self.HR(x4))))
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         signals = self.outc(x)
#         BVP_features = torch.squeeze(self.av(self.HR(x4)))
#         return signals, torch.squeeze(HR), BVP_features


class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        self.downsample = downsample
        self.Res = Res

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        return out


class rPPGNet(nn.Module):
    def __init__(self):
        super(rPPGNet, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.down = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlock(64, 64, 2, downsample=1),
            BasicBlock(64, 64, 1, downsample=1),
            BasicBlock(64, 128, 2, downsample=1),
            BasicBlock(128, 128, 1, downsample=1),
            BasicBlock(128, 256, 2, downsample=1, Res=1),
            BasicBlock(256, 256, 1, downsample=1),
            BasicBlock(256, 512, 2, downsample=1, Res=1),
            BasicBlock(512, 512, 1, downsample=1),
        )
        self.HR = nn.Sequential(BasicBlock(512, 256, 2, downsample=1, Res=1),
                                BasicBlock(256, 256, 1, downsample=0, Res=0),
                                BasicBlock(256, 512, 2, downsample=1, Res=1),
                                BasicBlock(512, 512, 1, downsample=0, Res=0)
                                )
        self.av = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 128, [2, 1], downsample=1),
            nn.ConvTranspose2d(128, 128, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(128, 64, [2, 1], downsample=1),
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 16, [2, 1], downsample=1),
            nn.ConvTranspose2d(16, 16, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(16, 1, [1, 1], downsample=1),
        )

    def forward(self, x):
        f = self.down(self.bn(x))
        rPPG = self.up(f).squeeze(dim=1)
        HR = self.fc(torch.squeeze(self.av(self.HR(f))))
        return rPPG, torch.squeeze(HR)


##################################### MLP SECTION DEPENDENCIES #######################################

# class MLP(nn.Module):
#   '''
#     Multilayer Perceptron for regression.
#   '''

#   def __init__(self):
#     super(MLP, self).__init__()
#     self.layers = nn.Sequential(
#         nn.Linear(640, 320),
#         nn.ReLU(),
#         # nn.Linear(1280, 2560),
#         # nn.ReLU(),
#         # nn.Linear(2560, 1280),
#         # nn.ReLU(),
#         # nn.Linear(1280, 640),
#         # nn.ReLU(),
#         # nn.Linear(640, 320),
#         # nn.ReLU(),
#         nn.Linear(320, 64),
#         nn.ReLU(),
#         nn.Linear(64, 1)
#     )

#   def forward(self, x):
#     '''
#       Forward pass
#     '''
#     return self.layers(x)



########################## COMBINED MODEL #########################


class TimeSformer_BVP(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        image_size=64,
        patch_size=16,
        channels=3,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.,
        n_channels=3, n_classes=1, bilinear=True        #BVP variables-----------------
    ):
        super(TimeSformer_BVP, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Embedding(num_positions + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),  # Time attention
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),  # Spatial attention
                # Feed Forward
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
            ]))

        ### gives HR prediction
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            # nn.Linear(dim,1)
        )

        ### gives tfr features...
        self.transformer_features = nn.LayerNorm(dim)

        # ------------------------------------------

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = unetPart.DoubleConv(n_channels, 48)
        self.down1 = unetPart.Down(48, 96)
        self.down2 = unetPart.Down(96, 192)
        self.down3 = unetPart.Down(192, 384)
        self.down4 = unetPart.Down(384, 384)
        self.up1 = unetPart.Up(768, 192, bilinear)
        self.up2 = unetPart.Up(384, 96, bilinear)
        self.up3 = unetPart.Up(192, 48, bilinear)
        self.up4 = unetPart.Up(96, 48, bilinear)
        self.outc = unetPart.OutConv(48, n_classes)
        self.HR = nn.Sequential(BasicBlock(384, 256, 2, downsample=1, Res=1),
                                BasicBlock(256, 256, 1, downsample=0, Res=0),
                                BasicBlock(256, 512, 2, downsample=1, Res=1),
                                BasicBlock(512, 512, 1, downsample=0, Res=0)
                                )
        self.av = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

        self.feature_reduction = nn.Sequential(nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 128))

        #-----------------------------------------
        self.mlp_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, video, x_Unet):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)
        video = rearrange(
            video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)

        tokens = self.to_patch_embedding(video)

        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        x = torch.cat((cls_token, tokens), dim=1)
        x += self.pos_emb(torch.arange(x.shape[1], device=device))

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f) + x
            x = ff(x) + x

        cls_token = x[:, 0]
        
        ## final tfr features...
        tfr_features = self.transformer_features(cls_token)             #--------new line added
        # print("tfr_features:- ", tfr_features.shape)
        # return self.to_out(cls_token)


        # ---------------------------------------------

        x1 = self.inc(x_Unet)                       #--------------var changed
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        HR = self.fc(torch.squeeze(self.av(self.HR(x4))))
        x_Unet = self.up1(x5, x4)                   #--------------var changed
        x_Unet = self.up2(x_Unet, x3)               #--------------var changed
        x_Unet = self.up3(x_Unet, x2)               #--------------var changed
        x_Unet = self.up4(x_Unet, x1)               #--------------var changed
        signals = self.outc(x_Unet)                 # -------------var changed
        
        ## final BVP features...
        # BVP_features = torch.squeeze(self.av(self.HR(x4)))          ## it will be of 512

        BVP_features_128 = self.feature_reduction(torch.squeeze(self.av(self.HR(x4))))  # it will be of 128
        # print("BVP SHAPE-->",BVP_features_128.shape)
        # return signals, torch.squeeze(HR)


        # -----------------------------------------------
        # concatenating both the features side by side.....

        # combinedFeatures = torch.cat((tfr_features, BVP_features_128),1)
        # print(combinedFeatures.shape)

        ## only for floops calculation....
        # BVP_features_128 = torch.reshape(BVP_features_128, (1,128))


        return signals, self.mlp_layers(torch.cat((tfr_features, BVP_features_128), 1))


