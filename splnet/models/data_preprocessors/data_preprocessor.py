from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import BaseDataPreprocessor
from torch import Tensor

from .scene_transform import PointTransform


@MODELS.register_module()
class FrameStackPreprocessor(BaseDataPreprocessor):
    """Project a sequence of point clouds into a stack of images.

    Args:
        H (int): Height of the 2D representation.
        W (int): Width of the 2D representation.
        fov_up (float): Front-of-View at upward direction of the sensor.
        fov_down (float): Front-of-View at downward direction of the sensor.
        ignore_index (int): The label index to be ignored.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
    """

    def __init__(self,
                 H: int,
                 W: int,
                 fov_up: float,
                 fov_down: float,
                 ignore_index: int,
                 non_blocking: bool = False) -> None:
        super().__init__(non_blocking=non_blocking)
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180 * np.pi
        self.fov_down = fov_down / 180 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index
        
        # transformation matrix for previous point clouds
        # self.trans1 = PointTransform()
        # self.trans2 = PointTransform()
        # self.trans3 = PointTransform()

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform spherical projection on each point cloud.
        Args:
            data (dict): Data from dataloader. The dict contains the whole
                batch data.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        # print(f'data in preprocessor: {data}')
        data.setdefault('data_samples', None)

        # print(data)

        inputs, data_samples = data['inputs'], data['data_samples']
        
        # a list of numbers, each of which is for a frame.
        self.offsets = data['offsets']
        
        # transform all the frames into a same coordinate system.
        # for points, offset in zip(inputs['points'], self.offsets):
        #     # print(f'points.shape: {points.shape}; offset: {offset}')
        #     points[:offset[0], :3] = self.trans1(points[:offset[0], :3])
        #     points[offset[0]:offset[1], :3] = self.trans2(points[offset[0]:offset[1], :3])
        #     points[offset[1]:, :3] = self.trans3(points[offset[1]:, :3])

        batch_inputs = dict()

        # assert 'points' in inputs
        # batch_inputs['points'] = inputs['points']
        
        # add 'semantic_seg' with shape of (H, W) into data_samples
        voxel_dict = self.frustum_region_group(inputs['points'], data_samples)
        # a list of offsets for each stacked frames.
        voxel_dict['offsets'] = data['offsets']
        batch_inputs['voxels'] = voxel_dict
        
        return {'inputs': batch_inputs, 'data_samples': data_samples}

    @torch.no_grad()
    def frustum_region_group(self, points: List[Tensor],
                             data_samples: SampleList) -> dict:
        """Calculate frustum region of each point.

        Args:
            points (List[Tensor]): Point cloud in one data batch.

        Returns:
            dict: Frustum region information.
        """
        voxel_dict = dict()
        
        # pixel coordinates of each point.
        coors = []
        # original points
        voxels = []
        # the unique voxel coordinates
        # voxel_coors = []
        # the inverse map of voxel coordinates to points
        # inverse_maps = []
        
        for i, res in enumerate(points):
            depth = torch.linalg.norm(res[:, :3], 2, dim=1)
            yaw = -torch.atan2(res[:, 1], res[:, 0])
            pitch = torch.arcsin(res[:, 2] / depth)

            coors_x = 0.5 * (yaw / np.pi + 1.0)
            coors_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

            # scale to image size using angular resolution
            coors_x *= self.W
            coors_y *= self.H

            # round and clamp for use as index
            coors_x = torch.floor(coors_x)
            coors_x = torch.clamp(coors_x, min=0, max=self.W - 1).type(torch.int64)

            coors_y = torch.floor(coors_y)
            coors_y = torch.clamp(coors_y, min=0, max=self.H - 1).type(torch.int64)
            
            # (N, 2)
            res_coors = torch.stack([coors_y, coors_x], dim=1)
            # (N, 3)
            # padding batch id before each pixel coordinates.
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            
            # padding frame id after each pixel coordinates.
            res_coors_frame_id = []
            begin_idx = 0
            for frame_id, offset in enumerate(self.offsets[i]):
                res_coors_frame_id.append(F.pad(res_coors[begin_idx:offset], (0, 1), mode='constant', value=frame_id))
                begin_idx = offset
            # (N, 4), i.e., [batch_id, x, y, frame_id]
            res_coors_frame_id = torch.cat(res_coors_frame_id, dim=0)
            
            coors.append(res_coors_frame_id)
            # coors.append(res_coors)
            voxels.append(torch.cat((res, depth[:, None], yaw[:, None], pitch[:, None]), dim=-1))

            if 'pts_semantic_mask' in data_samples[i].gt_pts_seg:
                import torch_scatter
                pts_semantic_mask = data_samples[i].gt_pts_seg.pts_semantic_mask
                seg_label = torch.ones(
                    (len(self.offsets[i]), self.H, self.W),
                    dtype=torch.long,
                    device=pts_semantic_mask.device) * self.ignore_index
                    
                # res_voxel_coors: the pixel coordinates of the image projected from the point cloud. 
                res_voxel_coors, inverse_map = torch.unique(
                    res_coors_frame_id, return_inverse=True, dim=0)
                
                # voxel_coors.append(res_voxel_coors)
                # inverse_maps.append(inverse_map)
                
                voxel_semantic_mask = torch_scatter.scatter_sum(
                    F.one_hot(pts_semantic_mask).float(), inverse_map, dim=0)
                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                # assign label for each pixel
                seg_label[res_voxel_coors[:, 3],
                          res_voxel_coors[:, 1],
                          res_voxel_coors[:, 2]] = voxel_semantic_mask
                          
                # the masks of the image.
                data_samples[i].gt_pts_seg.voxel_semantic_mask = seg_label
        
        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        # (B, H, W, F) -> (B, F, H, W)
        coors[:, [1, 2, 3]] = coors[:, [3, 1, 2]]
        
        
        # a batch of original points
        voxel_dict['pts'] = voxels
        # a batch of pixel coordinates of each point.
        voxel_dict['pts_coors'] = coors

        return voxel_dict
