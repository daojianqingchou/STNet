'''
Created on 2024年9月16日

@author: dongzi
'''

from typing import Sequence, Union, List, Text
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer, build_upsample_layer, ConvModule, MaxPool2d, Conv2d
from mmseg.models.backbones.unet import UNet


class ResidualImageModule(nn.Module):
    def __init__(self,
                 in_channels = 16,
                 base_channels = 16,
                 num_stages = 5,
                #  padding = [1, 2, 3],
                #  fusion_channels = [64, 32],
                 conv_cfg = dict(type='Conv2d'),
                 norm_cfg = dict(type='BN2d'),
                 act_cfg = dict(type='ReLU')
                 ):
        super().__init__()
        
        self.unet3 = UNet(in_channels=in_channels, base_channels=base_channels, num_stages=num_stages)

    def forward(self, x):
        r"""A multi-scale residual image module.
        params:
            x: (B, C, H, W)
        returns:
            A new learned feature map.
        """

        x =  self.unet3(x)  

        return x


class UNet(nn.Module):

    def __init__(self,
                 in_channels = 16,
                 base_channels = 16,
                 kernel_size = 3,
                 padding = 1,
                 num_stages = 5,
                 strides = [1, 1, 1, 1, 1],
                #  down_samples = [True, True, True, True, True],
                 upsample_cfg = dict(type='nearest', scale_factor=2),
                 conv_cfg = dict(type='Conv2d'),
                 norm_cfg = dict(type='BN2d'),
                 act_cfg = dict(type='ReLU')
                 ):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            self.encoder.append(nn.Sequential(
                ConvModule(in_channels, base_channels * 2**i, kernel_size, stride=strides[i], padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(base_channels * 2**i, base_channels * 2**i, kernel_size, stride=strides[i], padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                MaxPool2d(kernel_size=2, stride=2)
            ))

            self.decoder.append(nn.Sequential(
                build_upsample_layer(upsample_cfg),
                ConvModule(base_channels * 2**(i+1), base_channels * 2**i, kernel_size, stride=strides[i], padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(base_channels * 2**i, in_channels, kernel_size, stride=strides[i], padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ))

            in_channels = base_channels * 2**i

        self.decoder = self.decoder[::-1]

        self.middle = ConvModule(base_channels * 2**(num_stages-1), base_channels * 2**(num_stages-1), 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        enc_feats = []

        for i, enc in enumerate(self.encoder):
            x = enc(x)
            enc_feats.append(x)

        enc_feats = enc_feats[::-1]
        
        x = self.middle(x)

        for i, dec in enumerate(self.decoder):
            x = dec(torch.cat((x, enc_feats[i]), dim=1))

        return x

if __name__ == '__main__':
    x = torch.rand(2, 32, 64, 512)
    # x = get_res_img(x)

    backbone = ResidualImageModule(32, 16
                                   )

    x = backbone(x)

    print(x.shape)


    