'''
Created on 2024年10月15日

@author: Administrator
'''
from typing import Dict

import torch
from torch import Tensor

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptSampleList

@MODELS.register_module()
class SPLNet(EncoderDecoder3D):
    def __init__(self,
                 voxel_encoder,
                 backbone,
                 decode_head,
                 neck = None,
                 auxiliary_head = None,
                 train_cfg = None,
                 test_cfg = None,
                 data_preprocessor = None,
                 init_cfg = None
                 ):
        super().__init__(
            backbone = backbone,
            decode_head = decode_head,
            neck = neck,
            auxiliary_head = auxiliary_head,
            train_cfg = train_cfg,
            test_cfg = test_cfg,
            data_preprocessor = data_preprocessor,
            init_cfg = init_cfg)
    
        self.voxel_encoder = MODELS.build(voxel_encoder)
    
    
    def extract_feat(self, batch_inputs:Tensor) -> dict:
        """
        """
        # voxel_dict includes 'voxels', 'coors', and 'offsets'
        voxel_dict = batch_inputs['voxels'].copy()
        voxel_dict = self.voxel_encoder(voxel_dict)
        voxel_dict = self.backbone(voxel_dict)
        if self.with_neck:
            self.neck(voxel_dict)
        
        return voxel_dict
    
    
    def loss(self, batch_inputs_dict:dict, 
        batch_data_samples:SampleList)->Dict[str, Tensor]:
        
        voxel_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss_ce = self._decode_head_forward_train(voxel_dict, batch_data_samples)
        losses.update(loss_ce)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)
            
            
        return losses
    
    
    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(voxel_dict,
                                                   batch_input_metas,
                                                   self.test_cfg)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        return self.postprocess_result(seg_logits_list, batch_data_samples)
    
    
    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> dict:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            dict: Forward output of model without any post-processes.
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)


if __name__ == '__main__':
    pass