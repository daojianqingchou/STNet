'''
Created on 2024年9月19日

@author: gxd
'''

from typing import Optional, List, Union, Callable
import numpy as np
import torch

from mmdet3d.datasets import Seg3DDataset, SemanticKittiDataset, LoadPointsFromFile
from mmdet3d.registry import DATASETS

@DATASETS.register_module()
class SemantickittiMultiDataset(SemanticKittiDataset):
    
    METAINFO = {
        'classes': ('car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
                    'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
                    'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
                    'trunk', 'terrain', 'pole', 'traffic-sign', 'moving-car', 
                    'moving-bicyclist', 'moving-person', 'moving-motorcyclist',
                    'moving-other-vehicle', 'moving-truck'),
        'palette': [[100, 150, 245],  [100, 230, 245],   [30, 60, 150],
                    [80, 30, 180],    [100, 80, 250],    [155, 30, 30],
                    [255, 40, 200],   [150, 30, 90],     [255, 0, 255],
                    [255, 150, 255],  [75, 0, 75],       [175, 0, 75], 
                    [255, 200, 0],    [255, 120, 50],    [0, 175, 0], 
                    [135, 60, 0],     [150, 240, 80],    [255, 240, 150], 
                    [255, 0, 0],      [100, 150, 245],   [255, 40, 200],
                    [255, 30, 30],    [150, 30, 90],     [0, 0, 255], [80, 30, 180]],
        'seg_valid_class_ids':
        tuple(range(25)),
        'seg_all_class_ids':
        tuple(range(25)),
    }
    
    def __init__(self, 
        data_root:Optional[str]=None, 
        ann_file:str='', 
        metainfo:Optional[dict]=None, 
        data_prefix:dict=dict(
            pts='', 
            img='', 
            pts_instance_mask='', 
            pts_semantic_mask=''), 
        pipeline:List[Union[dict, Callable]] = [], 
        modality:dict = dict(use_lidar=True, use_camera=False), 
        ignore_index:Optional[int] = None, 
        scene_idxs:Optional[Union[str, np.ndarray]] = None, 
        test_mode:bool = False, 
        serialize_data:bool = False, 
        load_eval_anns:bool = True, 
        num_frames = 3,                                   # number of frames.
        backend_args:Optional[dict] = None, **
        kwargs)->None:
    
        self.num_frames = num_frames
        
        self.total_num = [
            0,
            4541,    # number of frames in scene 0.
            1101,    # 1
            4661,    # 2
            801,     # 3
            271,     # 4
            2761,    # 5
            1101,    # 6
            1101,    # 7
            4071,    # 8
            1591,    # 9
            1201,    # 10
            921,     # 11
            1061,    # 12
            3281,    # 13
            631,     # 14
            1901,    # 15
            1731,    # 16
            491,     # 17
            1801,    # 18
            4981,    # 19
            831,     # 20
            2721     # 21
        ]
        super().__init__(data_root = data_root, 
                         ann_file = ann_file, 
                         metainfo = metainfo, 
                         data_prefix=data_prefix, 
                         pipeline = pipeline, 
                         modality = modality, 
                         ignore_index = ignore_index, 
                         scene_idxs=scene_idxs, 
                         test_mode = test_mode, 
                         serialize_data=serialize_data, 
                         load_eval_anns=load_eval_anns, 
                         backend_args=backend_args, 
                         **kwargs)
    
    def get_total_num(self):
        return torch.cumsum(torch.as_tensor(self.total_num), 0)
    
    
    def prepare_data(self, idx:int)->dict:
        r"""
            Add previous 'self.frame_num-1' frames together.
        Args:
            idx: frame index
        Retures:
            dict: added ‘offset’.
            modified: 'points', 'pts_semantic_mask'.
        """
        # the index of current frame and previous frames, e.g., [6, 7, 8].
        frame_idx = torch.flip(torch.arange(self.num_frames), [0])    # [2, 1, 0]
        frame_idx = idx - frame_idx                     # e.g., 8 - [2, 1, 0] = [6, 7, 8]

        # get all point cloud frames processed by pipeline.
        results = []
        scene_border = self.get_total_num()
        
        for i, index in enumerate(frame_idx):
            if i > 0 and index in scene_border:
                # the sequence of frames cross two different scenes.
                # So we simply replace the previous frames in first scene with the starting frame in second scene.
                results.clear()
                j = 0
                while j <= i:
                    results.append(super().prepare_data(index))
                    j += 1
            else:
                results.append(super().prepare_data(index))
                
        # e.g., 8
        current_frame = results[-1]
        # the number of points in each frame.
        current_frame['offsets'] = [len(current_frame['inputs']['points'])]
        for i in range(2, len(results) + 1):
            # append the offset of previous frames.
            current_frame['offsets'].append(len(results[-i]['inputs']['points']))
            # concatenate points of previous frames and current frame together.
            current_frame['inputs']['points'] = torch.cat([results[-i]['inputs']['points'], current_frame['inputs']['points']], axis=0)
           
            if not self.test_mode:
                try:
                    # concatenate labels of previous frames and current frame together. 
                    current_frame['data_samples'].gt_pts_seg.pts_semantic_mask = torch.cat([results[-i]['data_samples'].gt_pts_seg.pts_semantic_mask, current_frame['data_samples'].gt_pts_seg.pts_semantic_mask], axis=0)
                except Exception as e:
                    print(f'---------------------------------------------------------------')
                    print(e)
                    print(f'current_frame: {current_frame}')
                    print(f'results: {results}')
                    print(f'===============================================================')
        # make the number of points in the same order as frames.
        current_frame['offsets'] = torch.cumsum(torch.flip(torch.as_tensor(current_frame['offsets']), [0]), 0)
        
        return current_frame

    
if __name__ == '__main__':
    ds = SemantickittiMultiDataset()
    print(ds.get_total_num())



