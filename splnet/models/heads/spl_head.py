'''
Created on 2024年10月15日

@author: Administrator
'''
from typing import Sequence, Dict, List

from mmdet3d.models.decode_heads.decode_head import Base3DDecodeHead
from mmdet3d.utils import ConfigType
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.registry import MODELS

import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import build_norm_layer, build_activation_layer

from easydict import EasyDict as edict
import einops as eo


@MODELS.register_module()
class SplHead(Base3DDecodeHead):
    def __init__(self,
                 in_channels: int,
                 middle_channels: Sequence[int],
                 moving_in_c: int,
                 moving_channels: Sequence[int],
                 voxel_in_c = 32,
                 voxel_channels = (32,),
                 num_frames: int = 3,
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg = dict(type='ReLU', inplace=True),
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                loss_bce = None,
                lovasz_loss = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_frames = num_frames

        self.learning_map_2 = torch.zeros((self.ignore_index + 1), dtype=torch.int64)
        self.learning_map_2[self.ignore_index] = self.ignore_index
        for i in range(19, 25):
            self.learning_map_2[i] = 1
        

        self.loss_ce = MODELS.build(loss_ce)

        if lovasz_loss:
            self.lovasz_loss = MODELS.build(lovasz_loss)
        else:
            self.lovasz_loss = None
        
        if loss_bce:
            self.loss_bce = MODELS.build(loss_bce)
        else:
            self.loss_bce = None

        # prediction head for point features.
        self.mlps = nn.ModuleList()
        for i, mid_channel in enumerate(middle_channels):
            self.mlps.append(nn.Sequential(
                nn.Linear(in_channels, mid_channel),
                build_norm_layer(norm_cfg, mid_channel)[1],
                build_activation_layer(act_cfg)
                ))
            in_channels = mid_channel
        
        # prediction head for moving features.
        self.moving_mlps = nn.ModuleList()
        for i, mov_c in enumerate(moving_channels):
            self.moving_mlps.append(nn.Sequential(
                nn.Linear(moving_in_c, mov_c),
                build_norm_layer(norm_cfg, mov_c)[1],
                build_activation_layer(act_cfg)
            ))
            moving_in_c = mov_c
        self.moving_seg = nn.Linear(moving_channels[-1], 2)

        # prediction head for voxel features.
        self.voxel_mlps = nn.ModuleList()
        for i, voxel_c in enumerate(voxel_channels):
            self.voxel_mlps.append(
                nn.Sequential(
                    nn.Linear(voxel_in_c, voxel_c),
                )
            )
            voxel_in_c = voxel_c
        self.voxel_seg = nn.Linear(middle_channels[-1], kwargs['num_classes'])


    def forward(self, voxel_dict:dict) -> dict:
        # print(f'voxel_dict: {voxel_dict.keys()}')
        voxel_dict = edict(voxel_dict)
        
        # point residual values of the current frame.
        # pts_res = voxel_dict.pts_res
        pts_res = voxel_dict.pts_feats_fused_cur_frame
        for i in range(len(self.moving_mlps)):
            pts_res = self.moving_mlps[i](pts_res)

        moving_logits = self.moving_seg(pts_res)
        voxel_dict.moving_logits = moving_logits

        # segmentation logits of the current frame.
        pts_feats_fused_cur_frame = voxel_dict.pts_feats_fused_cur_frame
        # fuse point features from point branch, voxel branch, and residual branch together.
        # pts_feats_fused_cur_frame = torch.cat((pts_feats_fused_cur_frame, pts_res), dim=-1)
        for i in range(len(self.mlps)):
            pts_feats_fused_cur_frame = self.mlps[i](pts_feats_fused_cur_frame)

        seg_logits_cur_frame = self.cls_seg(pts_feats_fused_cur_frame)
        voxel_dict.seg_logits_cur_frame = seg_logits_cur_frame
        # end of current frame.

        # logits of voxel
        voxel_feats = voxel_dict.out_dense
        for i in range(len(self.voxel_mlps)):
            voxel_feats = self.voxel_mlps[i](voxel_feats)
        
        seg_logits_voxel = self.voxel_seg(voxel_feats)
        voxel_dict.seg_logits_voxel = seg_logits_voxel
        # segmentation logiat of fll frames.
        # pts_feats_fused = voxel_dict.pts_feats_fused
        # for i in range(len(self.mlps)):
        #     pts_feats_fused = self.mlps[i](pts_feats_fused)
        
        # (N, classes_num)
        # seg_logits = self.cls_seg(pts_feats_fused)
        # voxel_dict.seg_logits = seg_logits
        # end of all frames.

        return voxel_dict


    def _stack_batch_gt(self, batch_data_samples:SampleList) -> torch.Tensor:
        # gather labels of a batch of point clouds into a list.
        gt_semantic_seg_list = [data_sample.gt_pts_seg.pts_semantic_mask 
                                for data_sample in batch_data_samples]
        
        # labels of a batch of grids.
        gt_voxel_semantic_list = [data_sample.gt_pts_seg.voxel_semantic_mask for data_sample in batch_data_samples]

        return torch.cat(gt_semantic_seg_list, dim=0), torch.stack(gt_voxel_semantic_list, dim=0)

    
    def loss_by_feat(self, voxel_dict: edict, 
        batch_data_samples:SampleList)->Dict[str, Tensor]:
        loss = edict()
        
        # seg_logits = voxel_dict.seg_logits
        pts_seg_label, voxel_seg_label = self._stack_batch_gt(batch_data_samples)
        # loss.loss_ce_pt = self.loss_ce(seg_logits, seg_label, ignore_index=self.ignore_index)


        # point labels of the current frame.
        pts_coors = voxel_dict.pts_coors
        seg_label_cur_frame = pts_seg_label[pts_coors[:, 1] == self.num_frames - 1]
        seg_logits_cur_frame = voxel_dict.seg_logits_cur_frame

        loss.loss_ce_pt = self.loss_ce(seg_logits_cur_frame, seg_label_cur_frame, ignore_index=self.ignore_index)
        # end of current frame.

        # calculating moving loss
        self.learning_map_2 = self.learning_map_2.to(device=seg_label_cur_frame.device)
        moving_label = self.learning_map_2[seg_label_cur_frame]
        moving_logits = voxel_dict.moving_logits

        if self.loss_bce:
            loss.loss_moving_ce_pt = self.loss_bce(moving_logits, moving_label, ignore_index = self.ignore_index)
        # end of moving loss.

        # calculating voxel loss
        seg_logits_voxel = voxel_dict.seg_logits_voxel
        seg_logits_voxel = eo.rearrange(seg_logits_voxel, 'b f h w c -> (b f h w) c')
        voxel_seg_label = eo.rearrange(voxel_seg_label, 'b f h w -> (b f h w)')
        loss.loss_voxel = self.loss_ce(seg_logits_voxel, voxel_seg_label, ignore_index = self.ignore_index)

        if self.lovasz_loss:
            # loss.loss_lovasz_pt = self.lovasz_loss(seg_logits, seg_label, ignore_index=self.ignore_index)

            # locasz loss of the current frame.
            loss.loss_lovasz_pt = self.lovasz_loss(seg_logits_cur_frame, seg_label_cur_frame, ignore_index=self.ignore_index)

            loss.loss_lovasz_voxel = self.lovasz_loss(seg_logits_voxel, voxel_seg_label, ignore_index = self.ignore_index)

            

        return loss
    
    
    def predict(self, voxel_dict:dict, batch_input_metas:List[dict], 
        test_cfg:ConfigType)->Tensor:

        voxel_dict = self.forward(voxel_dict)
        
        seg_pred_list = self.predict_by_feat(voxel_dict, batch_input_metas)
        
        # final_seg_pred_list = []
        # for seg_pred, input_metas in zip(seg_pred_list, batch_input_metas):
        #     if 'num_points' in input_metas:
        #         num_points = input_metas['num_points']
        #         seg_pred = seg_pred[:num_points]

        #     final_seg_pred_list.append(seg_pred)
        
        return seg_pred_list
        
        
    def predict_by_feat(self, voxel_dict: dict, batch_input_metas: List[dict]) -> List[Tensor]:
        """produce the list of predictions according to the logits.
        voxel_dict:
            'seg_logits': 
        """
        # seg_logits = voxel_dict['seg_logits']
        
        seg_pred_list = []
        offsets = voxel_dict['offsets']
        # (N, 4), [B, H, W, F]
        # pts_coors = voxel_dict['pts_coors']
        
        # the current frame
        seg_logits_cur_frame = voxel_dict.seg_logits_cur_frame
        start_idx = 0
        for i, offset in enumerate(offsets):
            end_idx = offset[-1] - offset[-2] + start_idx
            seg_pred_list.append(seg_logits_cur_frame[start_idx:end_idx])
            start_idx = end_idx
        
        return seg_pred_list
        
    
    def build_conv_seg(self, channels:int, num_classes:int, 
        kernel_size:int)->nn.Module:
        return nn.Linear(channels, num_classes)
    

if __name__ == '__main__':
    pass