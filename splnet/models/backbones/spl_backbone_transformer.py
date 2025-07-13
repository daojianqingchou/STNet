'''
Created on 2024年9月16日

@author: dongzi
'''
import math

from typing import Sequence, Union, List, Text
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer, ConvModule

import einops as eo

class MultiHeadAttentionModule(nn.Module):
    r"""
    A basic multi-head self-attention module.
    params:
        in_c: input channel
        out_c: output channel
        mid_c: middle channel for self-attention
        n_head; number of heads, default is 8.
        use_mask: using mask or not, default False.
        attn_dropout:
    """
    def __init__(self, in_c, out_c, 
                 mid_c = 64, 
                 w_size = 4, 
                 n_head = 8, 
                 pos_embedding = False,
                 attn_dropout = 0.2,
                 norm_cfg: ConfigType = dict(type='LN'),
                 act_cfg = dict(type='ReLU')
                 ):
        super().__init__()

        self.in_c = in_c
        self.mid_c = mid_c
        self.out_c = out_c        
        self.pos_embedding = pos_embedding

        if isinstance(w_size, int):
            self.H = w_size
            self.W = w_size
        elif isinstance(w_size, Sequence):
            self.H, self.W = w_size

        self.n_head = n_head

        if pos_embedding:
            self.fc_before = nn.Linear(in_c, mid_c)
        else:
            self.fc_before = nn.Linear(in_c, mid_c)
        self.norm_before = build_norm_layer(norm_cfg, mid_c)[1]
        self.act_before = build_activation_layer(act_cfg)

        # conditional position embedding
        if pos_embedding:
            self.conv_pos = nn.Conv3d(5, mid_c, 3, padding=1)

        # if pos_embedding:
        #     self.dict_pos_embedding = {}


        self.w_q = nn.Linear(mid_c, n_head * mid_c, bias=False)
        self.w_k = nn.Linear(mid_c, n_head * mid_c, bias=False)
        self.w_v = nn.Linear(mid_c, n_head * mid_c, bias=False)

        self.softmax = nn.Softmax(-1)
        self.dropout_mh = nn.Dropout(attn_dropout)
        self.dropout_ffl = nn.Dropout(attn_dropout)
            

        self.fc_mul_head = nn.Linear(n_head * mid_c, mid_c)
        self.norm_mh = build_norm_layer(norm_cfg, mid_c)[1]

        self.ffl = nn.Sequential(
            nn.Linear(mid_c, 4 * mid_c),
            build_activation_layer(act_cfg),
            nn.Linear(4 * mid_c, mid_c)
            )
        
        self.norm_ffl = build_norm_layer(norm_cfg, mid_c)[1]
            

        if mid_c != out_c:
            self.after_act_norm_fc = nn.Sequential(
                nn.Linear(mid_c, out_c),
                build_norm_layer(norm_cfg, out_c)[1],
                build_activation_layer(act_cfg)
            )
            

    def forward(self, x, mask = None):
        """
        params:
            x: (B, F, H, W, C)
        """
        B, F, H, W, C = x.shape

        # conditional position embedding
        if self.pos_embedding:
            pos_embedding = self.get_pos_embedding((B, F, H, W), self.H, self.W)
            pos_embedding = pos_embedding.to(device=x.device, dtype=torch.float)
            # pos_embedding
            # x = torch.cat((x, pos_embedding), dim=-1)


            # dynamic position embedding.
            pos_embedding = eo.rearrange(pos_embedding, 'b f h w c -> b c f h w')
            pos_embedding = self.conv_pos(pos_embedding)
            pos_embedding = eo.rearrange(pos_embedding, 'b c f h w -> b f h w c')
            
            x = x + pos_embedding

        # (B, H', W', N_V, C), H' and W' denotes number of windows along height and width
        # N_V = f x s_h x s_w denotes number of voxels in each window.
        x = eo.rearrange(x, 'b f (h s_h) (w s_w) c -> b h w (f s_h s_w) c', s_h=self.H, s_w=self.W)
        # in_c -> mid_d
        x = self.act_before(self.norm_before(self.fc_before(x)))

        # if mask != None:
        #     # (B, H', W', N_V, C) -> (B, H', W', N_V, 1)
        #     mask = torch.all(x == 0, dim=-1, keepdim=True).unsqueeze(3)

        x_q = self.w_q(x)
        # (B, H', W', N_V, C) -> (B, H', W', N_H, N_V, C), N_H denotes number of head.
        x_q = eo.rearrange(x_q, 'b h w n_v (n_h c) -> b h w n_h n_v c', n_h = self.n_head)

        x_k = self.w_k(x)
        # (B, H', W', N_V, C) -> (B, H', W', N_H, C, N_V), N_P denotes number of voxels
        x_k_T = eo.rearrange(x_k, 'b h w n_v (n_h c) -> b h w n_h c n_v', n_h = self.n_head)
        
        x_v = self.w_v(x)
        # (B, H', W', N_V, C) -> (B, H', W', N_H, N_V, C)
        x_v = eo.rearrange(x_v, 'b h w n_v (n_h c) -> b h w n_h n_v c', n_h = self.n_head)

        # (B, H', W', N_H, N_V, N_V)
        attn = torch.matmul(x_q, x_k_T / math.sqrt(x_q.shape[-1]))

        if mask != None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)
        # (B, H', W', N_H, N_V, C)
        v = torch.matmul(attn, x_v)

        # concatenate multi-head features.
        # (B, H', W', N_H, N_V, C) -> (B, H', W', N_V, C)
        v = eo.rearrange(v, 'b h w n_h n_v c -> b h w n_v (n_h c)')

        # add & norm
        x = self.norm_mh(self.dropout_mh(self.fc_mul_head(v)) + x)

        # add & norm
        x = self.norm_ffl(self.dropout_ffl(self.ffl(x)) + x)

        if self.mid_c != self.out_c:
            x = self.after_act_norm_fc(x)

        # (B, H', W', N_V, C) -> (B, F, H, W, C)
        x = eo.rearrange(x, 'b h w (f s_h s_w) c -> b f (h s_h) (w s_w) c', s_h=self.H, s_w=self.W)

        return x


    def get_abs_pos(self, shape):
        r"""
        params:
            shape: defaut (F, H, W), F is in ascendence, e.g. (-2, -1, 0).
        returns:
            pos: absolute position embedding with shpa of (F, H, W, 3)
        """
        F, H, W = shape
        
        f = torch.arange(F)
        f -= torch.max(f)

        h = torch.arange(H)
        w = torch.arange(W)

        # (F, H, W, 3)
        pos = torch.stack(torch.meshgrid(f, h, w, indexing='ij'), dim=-1)

        return pos

    def get_rel_pos(self, s_h, s_w):
        
        h = torch.arange(s_h)
        w = torch.arange(s_w)

        pos = torch.stack(torch.meshgrid(h, w, indexing='ij'), dim=-1)

        return pos
    

    def get_pos_embedding(self, shape, s_h, s_w):
        B, F, H, W = shape

        key_pos = ','.join((str(B), str(F), str(H), str(W)))
        if key_pos in self.dict_pos_embedding:
            return self.dict_pos_embedding[key_pos]

        # (F, H, W, 3)
        abs_pos = self.get_abs_pos((F, H, W))
        rel_pos = self.get_rel_pos(s_h, s_w)

        copy_h = H // s_h
        copy_w = W // s_w

        rel_pos = eo.repeat(rel_pos, 'h w c -> f (c_h h) (c_w w) c', f=F, c_h=copy_h, c_w=copy_w)

        pos_embedding = torch.cat((abs_pos, rel_pos), dim=-1)
        # (B, F, H, W, C)
        pos_embedding = eo.repeat(pos_embedding, 'F H W C -> b F H W C', b = B)

        self.dict_pos_embedding[key_pos] = pos_embedding

        return pos_embedding



    def get_mask_sparse(self, sp):
        indices = sp.indices
        batch_size = sp.batch_size
        spatial_shape = sp.spatial_shape
        mask_shape = [batch_size] + spatial_shape
        mask = torch.zeros(mask_shape, device=indices.device)
        mask[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = 1

        # (B, F, H, W) -> (B, H', W', 1, 1, N_V)
        mask = eo.rearrange(mask, 'b f (h s_h) (w s_w) -> b h w 1 1 (f s_h s_w)', s_h=self.H, s_w=self.W)

        # make the mask a symmetric matrix
        # num_seq = self.H * self.W * spatial_shape[-1]
        # (B, H, W, F) -> (B, H', W', 1, N_V, N_V)
        # mask = eo.repeat(mask, 'b (h s_h) (w s_w) f -> b h w 1 (s_h s_w f) n', s_h=self.H, s_w=self.W, n=num_seq)
        
        mask = mask == 0

        # mask_T = mask.transpose(-1, -2)
        # mask = torch.logical_or(mask, mask_T)

        return mask

    
if __name__ == '__main__':
    x = torch.rand(2, 3, 64, 512, 32)
    trans = MultiHeadAttentionModule(32, 64, mid_c=128)

    print(trans.get_pos_embedding((2, 3, 64, 512), 8, 8).shape)


    # x = trans(x)

    # print(x.shape)
    