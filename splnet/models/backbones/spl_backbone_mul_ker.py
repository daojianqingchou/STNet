'''
Created on 2024年9月16日

@author: dongzi
'''
from typing import Sequence, Union, List, Text
import torch
import torch.nn as nn
import numpy as np

from mmdet3d.registry import MODELS
# from mmengine.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer, ConvModule

import torch_scatter
import spconv

from .spl_backbone_transformer import MultiHeadAttentionModule
from .residual_img import ResidualImageModule

from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    print('spconv2 is available.')
    from spconv.pytorch import SparseConv3d, SparseInverseConv3d, SubMConv3d, SparseConvTensor, SparseSequential
    from spconv.pytorch import functional as Fsp
else:
    print('spconv is available.')
    from mmcv.ops import SparseConvTensor, SparseConv3d, SparseInverseConv3d, SubMConv3d, SparseSequential

@ MODELS.register_module()
class FrameStackBackbone(BaseModule):
    def __init__(self, 
                 in_channels: int,
                 # point_in_channels: int,
                 # output_shape: Sequence[int],
                 H = 64,
                 W = 512,
                 num_frames = 3,
                 # depth: int,
                 # ignore_index = 255,
                 trans_c = 64,
                 w_sizes = (4),
                 pos_embedding = False,
                 attn_dropout = 0.2,
                 out_channels: Sequence[int] = (64, 128, 512, 1024),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 # fuse_channels: Sequence[int] = (256, 128),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 # point_norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg:Union[dict, List[dict], None]=None) -> None:
        BaseModule.__init__(self, init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.H = H
        self.W = W
        self.num_frames = num_frames
        self.strides = strides
        self.cum_strides = np.cumprod(strides, -1).tolist()
        self.inv_strides = self.cum_strides[::-1]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
        # sparse convolution blocks used to extract features
        self.sp_conv_list = nn.ModuleList()
        self.sp_deconv_list = nn.ModuleList()
        self.sp_conv_fuse_list = nn.ModuleList()


        self.sp_conv_list_5 = nn.ModuleList()
        self.sp_deconv_list_5 = nn.ModuleList()
        self.sp_deconv_fuse_list = nn.ModuleList()

        # transformer blocks
        self.down_trans_list = nn.ModuleList()
        self.up_trans_list = nn.ModuleList()

        # point blocks
        # self.enc_point_list = nn.ModuleList()
        # self.dec_point_list = nn.ModuleList()
        
        # middle of the UNet.
        self.middle_spconv = SparseSequential(
            SubMConv3d(out_channels[-1], 
                       out_channels[-1],
                       3,
                       indice_key = f'stage_{len(out_channels)}'),
            build_norm_layer(dict(type='BN1d', eps=1e-5, momentum=0.1), out_channels[-1])[1],
            build_activation_layer(dict(type='LeakyReLU')),
            
            )
        
        # self.pre_point_module = self._make_point_layer(self.in_channels, out_channels[1])
        # self.middle_point_module = self._make_point_layer(out_channels[1], out_channels[-1])
        # self.after_point_module = self._make_point_layer(out_channels[-1], self.in_channels)


        # build a UNet-like backbone to extract spatial-temporal features of each voxel.
        m_in = self.in_channels
        for i, m_out in enumerate(out_channels):
            # down stream
            self.sp_conv_list.append(self._make_sparse_layer(m_in, m_out, 3, stride=strides[i], indice_key=f'stage3_{i}'))
            self.sp_conv_list_5.append(self._make_sparse_layer(m_in, m_out, 5, stride=strides[i], indice_key=f'stage5_{i}'))
            self.down_trans_list.append(MultiHeadAttentionModule(m_out*2, m_out, mid_c=trans_c, w_size=w_sizes[i], pos_embedding=pos_embedding, attn_dropout=attn_dropout))

            # up stream
            self.sp_deconv_list.append(self._make_de_sparse_layer(m_out*2, m_in, 3, stride=strides[i], indice_key=f'stage3_{i}'))
            self.sp_deconv_list_5.append(self._make_de_sparse_layer(m_out*2, m_in, 5, stride=strides[i], indice_key=f'stage5_{i}'))
            self.up_trans_list.append(MultiHeadAttentionModule(m_in*2, m_in, mid_c=trans_c, w_size=w_sizes[i], pos_embedding=pos_embedding, attn_dropout=attn_dropout))

            m_in = m_out


        # transpose the deconvolusion module list to match the data.
        self.sp_deconv_list = self.sp_deconv_list[::-1]
        self.sp_deconv_list_5 = self.sp_deconv_list_5[::-1]
        self.up_trans_list = self.up_trans_list[::-1]
        # self.dec_point_list = self.dec_point_list[::-1]
        # end of the UNet
            
        # residual image branch
        self.residual_branch = ResidualImageModule(
            in_channels = 32,
        )
    
    
    def _make_sparse_layer(self,
                          inplanes: int,
                          planes: int,
                          kernel_size: int = 3,
                          stride: int = 1,
                          padding: int = 1,
                          indice_key: Text = None
                          ):
        """
        build a basic sparse conv layer.
        """
        return SparseSequential(
            SubMConv3d(inplanes,
                       inplanes,
                       kernel_size,
                       stride = 1,
                       padding = kernel_size // 2,
                       indice_key = indice_key
                ),
            build_norm_layer(dict(type='BN1d', eps=1e-5, momentum=0.1), inplanes)[1],
            build_activation_layer(dict(type='LeakyReLU')),
            
            SubMConv3d(inplanes, 
                       planes,
                       kernel_size,
                       stride = stride,
                        padding = kernel_size // 2, 
                        indice_key = indice_key,
                        bias=False
                        ) 
            if stride == 1 else 
            SparseConv3d(inplanes,
                        planes, 
                        (3, 2, 2),                  # (F, H, W)
                        stride = (1, 2, 2),
                        padding = (1, 0, 0),
                        indice_key = '_'.join((indice_key, 'down')),
                        bias=False),
            build_norm_layer(dict(type='BN1d', eps=1e-5, momentum=0.1), planes)[1],
            build_activation_layer(dict(type='LeakyReLU')),
            
            SubMConv3d(planes, 
                       planes, 
                       kernel_size,
                       stride = 1,
                       padding = kernel_size // 2, 
                       # if stride is larger than 1, the indice_key would be made for next layer.
                       indice_key = indice_key if stride == 1 else ''.join((indice_key[:indice_key.index('_')+1], 
                                                        str(int(indice_key[indice_key.index('_')+1:])+1))),
                       bias=False),
            build_norm_layer(dict(type='BN1d', eps=1e-5, momentum=0.1), planes)[1],
            build_activation_layer(dict(type='LeakyReLU'))
            )
        
    
    def _make_de_sparse_layer(self,
                              inplanes: int,
                              planes: int,
                              kernel_size: int,
                              stride: int,
                              indice_key: Text = None):
        return SparseSequential(
            SubMConv3d(inplanes,
                      planes,
                      kernel_size,
                      indice_key = indice_key) if stride == 1 else
            SparseInverseConv3d(inplanes,
                                planes, 
                                (3, 2, 2),              # (F, H, W)
                                '_'.join([indice_key, 'down'])),
            build_norm_layer(dict(type='BN1d', eps=1e-5, momentum=0.1), planes)[1],
            build_activation_layer(dict(type='LeakyReLU')),
            
            
            SubMConv3d(planes, 
                       planes, 
                       kernel_size, 
                       padding = kernel_size // 2, 
                       indice_key = indice_key # ''.join((indice_key[:-1], str(int(indice_key[-1]) - 1))) if (int(indice_key[-1]) - 1) >= 0 else 0
                       ),
            build_norm_layer(dict(type='BN1d', eps=1e-5, momentum=0.1), planes)[1],
            build_activation_layer(dict(type='LeakyReLU')),
            )
    
    
    def _make_point_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Linear(in_c, out_c),
            build_norm_layer(dict(type='BN1d', eps=1e-5, momentum=0.1), out_c)[1],
            build_activation_layer(dict(type='ReLU', inplace=True))
        )

    def forward(self, voxel_dict: dict) -> dict:
        # (N, C)
        # point_feats = voxel_dict['point_feats'][-1]
        point_feats = voxel_dict['point_feats']
        # (N', C)
        voxel_feats = voxel_dict['voxel_feats']
        # (N', 4), [batch_id, frame_id, h, w]
        voxel_coors = voxel_dict['voxel_coors']
        # pixel coors of each point. (N, 4), [batch_id, frame_id, h, w]
        pts_coors = voxel_dict['pts_coors']
        
        spatial_shape = (self.num_frames, self.H, self.W)
        
        batch_size = voxel_coors[-1, 0].item() + 1
        sp_stack_frame = SparseConvTensor(voxel_feats, voxel_coors.to(torch.int32), spatial_shape, batch_size)
        
        # point_feats = self.pre_point_module(point_feats)
        

        # output of each conv block.
        out_spconv_list = [sp_stack_frame]
        x = sp_stack_frame
        for i, sp_conv in enumerate(self.sp_conv_list):
            x_3 = sp_conv(x)
            x_5 = self.sp_conv_list_5[i](x)

            # fuse the features from the two sparse conv blocks.
            x = self.cat_sparse_tensor(x_3, x_5)

            # apply a self-attention block
            x_d = x.dense(channels_first=False)
            mask = self.down_trans_list[i].get_mask_sparse(x)
            x_tran = self.down_trans_list[i](x_d, mask)

            x = self.dense2sparse(x_tran, x)

            # save the value for using in up sampling.
            out_spconv_list.append(x)

        x = self.middle_spconv(x)
        # point_feats = self.middle_point_module(point_feats)

        # output of each deconv block.
        out_spdeconv_list = []
        for i, sp_deconv in enumerate(self.sp_deconv_list):
            x = self.cat_sparse_tensor(x, out_spconv_list[-i-1])
            
            x_3 = sp_deconv(x)
            x_5 = self.sp_deconv_list_5[i](x)
            x = self.cat_sparse_tensor(x_3, x_5)

            # apply a self-attention block
            x_d = x.dense(channels_first=False)
            mask = self.up_trans_list[i].get_mask_sparse(x)
            x_tran = self.up_trans_list[i](x_d, mask)
            

            if i < len(self.sp_deconv_list) - 1:
                x = self.dense2sparse(x_tran, x)
            else:
                # # (B, H, W, F, C)
                x = x_tran

        # used if transformer is droped.
        # x = x.dense(channels_first=False)
        # (B, F, H, W, C)
        voxel_dict['out_dense'] = x

        # point features after post processing.
        # point_feats = self.after_point_module(point_feats)

        # (N, C)
        pts_feats_voxel = x[pts_coors[:, 0], pts_coors[:, 1], pts_coors[:, 2], pts_coors[:, 3]]

        # pseudo image with shape of (B, 2C, H, W)
        residual_img = voxel_dict['residual_img']
        # residual image branch (B, 2C, H, W)
        residual_img = self.residual_branch(residual_img)

        # the point coordinates of the current frame, (N, 4), [b, f, x, y].
        pts_coors_cur_frame = pts_coors[pts_coors[:, 1] == (self.num_frames - 1)]

        #######################
        # point features of the current frame. ############
        pts_feats_cur_frame = point_feats[pts_coors[:, 1] == (self.num_frames - 1)]
        # point features of the current frame from voxels. ##############
        pts_feats_voxel_cur_frame = x[pts_coors_cur_frame[:, 0], pts_coors_cur_frame[:, 1], pts_coors_cur_frame[:, 2], pts_coors_cur_frame[:, 3]]
        

        pts_res = residual_img[pts_coors_cur_frame[:, 0], :, pts_coors_cur_frame[:, 2], pts_coors_cur_frame[:, 3]]
        # (N, 3C)
        voxel_dict['pts_res'] = pts_res

        # fuse multiple point features of current frame together.
        pts_feats_fused_cur_frame = torch.cat([pts_feats_voxel_cur_frame, pts_feats_cur_frame, pts_res], dim=-1)
        voxel_dict['pts_feats_fused_cur_frame'] = pts_feats_fused_cur_frame
        
        # fuse multiple point features together.
        # pts_feats_fused = torch.cat([pts_feats_voxel, point_feats], dim=-1)
        # voxel_dict['pts_feats_fused'] = pts_feats_fused

        
        return voxel_dict
    

    def voxel2point(self, voxel, pts_coors, strides):
        r""" Transform features from voxel to points.
        params:
            voxel: voxel features with shape of (B, H, W, F, C)
            pts_coors: voxel coordinates for each point, (N, 4)
            strides: int or list of int.
        """
        if isinstance(strides, int):
            pts_feats = voxel[pts_coors[:, 0], pts_coors[:, 1] // strides, pts_coors[:, 2] // strides, pts_coors[:, 3]]
        elif isinstance(strides, (list, tuple)):
            pts_feats = voxel[pts_coors[:, 0], pts_coors[:, 1] // strides[0], pts_coors[:, 2] // strides[1], pts_coors[:, 3]]

        return pts_feats
    
    def point2voxel(self, pts_feats, pts_coors, shape, strides):
        r""" Transform features from points to voxel.
        params:
            pts_feats: (N, C)
            pts_coors: (N, 4)
            strides: int or list of int
        """
        pts_coors_stride = torch.stack([pts_coors[:, 0], pts_coors[:, 1] // strides, pts_coors[:, 2] // strides, pts_coors[:, 3]], dim=-1)
        if isinstance(strides, int):
            voxel_coors, inverse_map = torch.unique(pts_coors_stride, return_inverse=True, dim=0)
        elif isinstance(strides, (list, tuple)):
            voxel_coors, inverse_map = torch.unique(pts_coors_stride, return_inverse=True, dim=0)
        
        voxel_feats = torch_scatter.scatter_max(pts_feats, inverse_map, dim=0)[0]

        voxel = torch.zeros(shape, device=voxel_coors.device)
        voxel[voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2], voxel_coors[:, 3]] = voxel_feats

        return voxel

    
    def dense2sparse(self, dense, sparse):
        r""" Transform features from the dense tensor to the sparse tensor according to
        the indices of the sparse tensor.
        params:
            dense: A tensor feature with shape of (B, F, H, W, C).
            sparse: A sparse tensor
        """
        indices = sparse.indices
        feats = dense[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        return sparse.replace_feature(feats)


    def cat_sparse_tensor(self, a, b):
        """
        Args:
            a, b: the sparse tensors with same spatial shape.
        Returns:
            A fused sparse tensor.
        """
        assert a.spatial_shape == b.spatial_shape, f'a {a.spatial_shape} and b {b.spatial_shape} must have the same shape and features'
        
        # Test if a and b have the same shape and nonzero pisitions. 
        if torch.equal(a.indices, b.indices):
            feats = torch.cat([a.features, b.features], dim=-1)
            indices = a.indices

        else:
            # (B, F, H, W, C) 
            a_dense = a.dense(channels_first=False)
            b_dense = b.dense(channels_first=False)
            fused_dense = torch.cat([a_dense, b_dense], dim=-1)
            
            indices = torch.nonzero(torch.any(fused_dense!=0, dim=-1)).to(torch.int32)
            feats = fused_dense[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
        
        # merge the indice dictionary of the two sparse tensors.
        indice_dict_a = a.indice_dict
        indice_dict_b = b.indice_dict
        indice_dict = {**indice_dict_a, **indice_dict_b}

        return SparseConvTensor(feats, indices, spatial_shape=a.spatial_shape, batch_size=a.batch_size, indice_dict=indice_dict)
        
    
if __name__ == '__main__':
    pass