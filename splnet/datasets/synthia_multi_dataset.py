'''
Created on 2024年9月19日

@author: gxd
'''

from typing import Optional, List, Union, Callable
import numpy as np
import torch

from mmdet3d.datasets import Seg3DDataset, SemanticKittiDataset, LoadPointsFromFile
from mmdet3d.registry import DATASETS

nan = np.nan

@DATASETS.register_module()
class SynthiaMultiDataset(Seg3DDataset):
    
    METAINFO = {
        'classes': ('Building', 'Road', 'Sidewalk', 'Fence', 'Vegetation', 
                    'Pole',' Car', 'Traffic Sign', 'Pedestrian', 'Bicycle', 
                    'Lane Marking', 'Traffic Light'), 
        'palette': [[128, 0,   0], [128, 64,  128], [0,   0,   192], [64,  64,  128], [128, 128, 0], 
                    [192, 192, 128], [64,  0,   128], [192, 128, 128], [64,  64,  0], [0,   128, 192], 
                    [0,   172, 0], [0,   128, 128]], 
        'seg_valid_class_ids':
        tuple(range(12)),
        'seg_all_class_ids':
        tuple(range(12)),
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
        backend_args:Optional[dict] = None, 
        ** kwargs)->None:
    
        self.num_frames = num_frames
        
        self.total_num = [
            0,
            1451,    # number of frames in SEQ01 DAWN.
            1171,    # SEQ01 FALL
            935,     # SEQ01 NIGHT
            944,     # SEQ01 SUMMER
            1027,    # SEQ01 WINTER
            964,     # SEQ01 WINTERNIGHT
            941,     # seq02 DAWN
            742,     # seq02 FALL
            719,     # SEQ02 NIGHT
            823,     # SEQ02 RAINNIGHT
            1004,    # SEQ02 SOFTRAIN
            888,     # SEQ02 SUMMER
            1024,    # SEQ02 WINTER
            850,     # SEQ04 DAWN
            911,     # SEQ04 FALL
            813,     # SEQ04 NIGHT
            1115,    # SEQ04 RAINNIGHT
            936,     # SEQ04 SOFTRAIN
            901,     # SEQ04 SUMMER
            947,     # SEQ04 WINTER
            785,     # SEQ04 WINTERNIGHT
            815,     # SEQ05 FOG
            1045,    # SEQ06 SPRING
            842      # SEQ06 SUNSET
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

        if isinstance(current_frame['inputs'], List):
            current_frame['offsets'] = []
            for i in range(len(current_frame['inputs'])):
                # i means i-th augmented data.
                current_frame['offsets'].append([len(current_frame['inputs'][i]['points'])])
                for f in range(2, len(results) + 1):
                    # append the offset of previous frames.
                    current_frame['offsets'][i].append(len(results[-f]['inputs'][i]['points']))
                    # concatenate points of previous frames and current frame together.
                    current_frame['inputs'][i]['points'] = torch.cat([results[-f]['inputs'][i]['points'], current_frame['inputs'][i]['points']], axis=0)
                
                    current_frame['data_samples'][i].gt_pts_seg.pts_semantic_mask = torch.cat([results[-f]['data_samples'][i].gt_pts_seg.pts_semantic_mask,
                                                                                       current_frame['data_samples'][i].gt_pts_seg.pts_semantic_mask], axis=0)
                
                current_frame['offsets'][i] = torch.cumsum(torch.flip(torch.as_tensor(current_frame['offsets'][i]), [0]), 0)
        else:
            current_frame['offsets'] = [len(current_frame['inputs']['points'])]
            for i in range(2, len(results) + 1):
                # append the offset of previous frames.
                current_frame['offsets'].append(len(results[-i]['inputs']['points']))
                # concatenate points of previous frames and current frame together.
                current_frame['inputs']['points'] = torch.cat([results[-i]['inputs']['points'], current_frame['inputs']['points']], axis=0)
            
                current_frame['data_samples'].gt_pts_seg.pts_semantic_mask = torch.cat([results[-i]['data_samples'].gt_pts_seg.pts_semantic_mask, current_frame['data_samples'].gt_pts_seg.pts_semantic_mask], axis=0)
            
            # make the number of points in the same order as frames.
            current_frame['offsets'] = torch.cumsum(torch.flip(torch.as_tensor(current_frame['offsets']), [0]), 0)
        
        return current_frame


    def get_seg_label_mapping(self, metainfo=None):
        seg_label_mapping = np.zeros(metainfo['max_label']+1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]

        return seg_label_mapping
    
if __name__ == '__main__':
    ds = SynthiaMultiDataset()
    print(ds.get_total_num())



