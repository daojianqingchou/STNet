from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch_scatter
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType

from einops import rearrange

@MODELS.register_module()
class FrameStackEncoder(nn.Module):
    """Frustum Feature Encoder.

    Args:
        in_channels (int): Number of input features, either x, y, z or
            x, y, z, r. Defaults to 4.
        feat_channels (Sequence[int]): Number of features in each of the N
            FFELayers. Defaults to [].
        with_distance (bool): Whether to include Euclidean distance to points.
            Defaults to False.
        with_cluster_center (bool): Whether to include cluster center.
            Defaults to False.
        norm_cfg (dict or :obj:`ConfigDict`): Config dict of normalization
            layers. Defaults to dict(type='BN1d', eps=1e-5, momentum=0.1).
        with_pre_norm (bool): Whether to use the norm layer before input ffe
            layer. Defaults to False.
        feat_compression (int, optional): The frustum feature compression
            channels. Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = [],
                 with_voxel_coors = False,
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                #  in_shape: Sequence[int] = (64, 512),   # (H, W)
                 H = 64,
                 W = 512,
                 ignore_index = 255,
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-5, momentum=0.1),
                 with_pre_norm: bool = False,
                 feat_compression: Optional[int] = None) -> None:
        super().__init__()
        assert len(feat_channels) > 0

        self.H, self.W = H, W
        self.ignore_index = ignore_index

        if with_voxel_coors:
            in_channels += 4
        if with_distance:
            in_channels += 1
        if with_cluster_center:
            in_channels += 3

        self.with_voxel_coors = with_voxel_coors
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center

        feat_channels = [self.in_channels] + list(feat_channels)
        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1]
        else:
            self.pre_norm = None

        ffe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                ffe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                ffe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters, bias=False),
                        norm_layer, nn.ReLU(inplace=True)))
        self.ffe_layers = nn.ModuleList(ffe_layers)

        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression),
                nn.ReLU(inplace=True))

    def forward(self, voxel_dict: dict) -> dict:
        """
        Args:
            voxel_dict: including 'pts', 'pts_coors'.
                'pts': original points coordinates. (N, 3)
                'pts_coors': pixel coordinates of each point. (N, 4), [batch_id, frame_id, h, w]
                offsets:
        Returns:
            voxel_dict: add 'voxel_coors', 'voxel_feats', and 'point_feats'.
                'voxel_coors': (N, 4), [batch_id, framd_id, h, w] 
                    unique pixel coordinates.
                'voxel_feats':  (N, C), has same number of 'voxel_coors'
                    pixel features obtained by gathering features of all points falled into this pixel.
                'point_feats': point features including coordinates, distance, relative coordinates, and learned features.
                'stack_frame': stack a sequence of frames together with shape of (B, C, F, H, W)
        """
        # a batch of original points. (N, 4), [x, y, z, r]
        features = voxel_dict['pts']
        # a batch of pixel coordinates, e.g., [batch_id, frame_id, h, w]
        coors = voxel_dict['pts_coors']

        features_ls = [features]

        if self.with_voxel_coors:
            features_ls.append(coors)

        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Find distance of x, y, and z from frustum center
        if self._with_cluster_center:
            voxel_mean = torch_scatter.scatter_mean(
                features, inverse_map, dim=0)
            points_mean = voxel_mean[inverse_map]
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        # point_feats = []
        point_feats = None
        for ffe in self.ffe_layers:
            features = ffe(features)
            # point_feats.append(features)
        point_feats = features

        voxel_feats = torch_scatter.scatter_max(
            features, inverse_map, dim=0)[0]

        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)
        
        # construct regular stacked image.
        batch_size, frame_size = coors[-1, 0] + 1, coors[-1, 1] + 1
        feat_channel = voxel_feats.shape[-1]
        # (B, F, H, W, C)
        stack_frame = torch.zeros((batch_size, frame_size, self.H, self.W, feat_channel), device=voxel_coors.device)
        stack_frame[voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2], voxel_coors[:, 3]] = voxel_feats
        #
        # stack_frame: (B, H, W, F, C)
        # voxel_dict['stack_frame'] = stack_frame

        img_frames = []
        for f in range(frame_size - 1):
            # (B, H, W, C)
            img_frames.append(stack_frame[:, -1] - stack_frame[:, f])

        # (B, H, W, F*C)
        if img_frames:
            residual_img = torch.cat(img_frames, dim=-1)
        else:
            residual_img = stack_frame[:, -1]

        # (B, H, W, C) -> (B, C, H, W)
        residual_img = rearrange(residual_img, 'b h w c -> b c h w')
        voxel_dict['residual_img'] = residual_img
        
        
        voxel_dict['voxel_feats'] = voxel_feats
        voxel_dict['voxel_coors'] = voxel_coors
        voxel_dict['point_feats'] = point_feats
        
        
        return voxel_dict
