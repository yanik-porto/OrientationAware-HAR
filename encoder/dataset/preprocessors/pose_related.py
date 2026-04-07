import numpy as np
import random
import math
import torch
from typing import Dict

from .utils import Compose

class PreNormalize2D:
    """Normalize the range of keypoint values. """

    def __init__(self, img_shape=(1080, 1920), threshold=0.01, mode='fix', concatenate=True):
        self.threshold = threshold
        # Will skip points with score less than threshold
        self.img_shape = img_shape
        self.mode = mode
        self.concatenate = concatenate
        assert mode in ['fix', 'auto', 'auto_seq']

    def __call__(self, results):
        mask, maskout, keypoint_score, keypoint,  = None, None, None, results['keypoint'].astype(np.float32)

        if 'keypoint_score' in results:
            keypoint_score = results.pop('keypoint_score').astype(np.float32)
            if self.concatenate:
                keypoint = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)
                
        if keypoint.shape[-1] == 3:
            mask = keypoint[..., 2] > self.threshold
            maskout = keypoint[..., 2] <= self.threshold
        elif keypoint_score is not None:
            mask = keypoint_score > self.threshold
            maskout = keypoint_score <= self.threshold

        if self.mode == 'auto_seq':
            for im in range(len(keypoint)):
                if mask is not None:
                    if np.sum(mask):
                        x_max, x_min = np.max(keypoint[mask, 0]), np.min(keypoint[mask, 0])
                        y_max, y_min = np.max(keypoint[mask, 1]), np.min(keypoint[mask, 1])
                    else:
                        x_max, x_min, y_max, y_min = 0, 0, 0, 0
                else:
                    x_max, x_min = np.max(keypoint[im, :, :, 0]), np.min(keypoint[im, :, :, 0])
                    y_max, y_min = np.max(keypoint[im, :, :, 1]), np.min(keypoint[im, :, :, 1])
                if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                    keypoint[im, :, :, 0] = (keypoint[im, :, :, 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
                    keypoint[im, :, :, 1] = (keypoint[im, :, :, 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2
        elif self.mode == 'auto':
            if mask is not None:
                if np.sum(mask):
                    x_max, x_min = np.max(keypoint[mask, 0]), np.min(keypoint[mask, 0])
                    y_max, y_min = np.max(keypoint[mask, 1]), np.min(keypoint[mask, 1])
                else:
                    x_max, x_min, y_max, y_min = 0, 0, 0, 0
            else:
                x_max, x_min = np.max(keypoint[..., 0]), np.min(keypoint[..., 0])
                y_max, y_min = np.max(keypoint[..., 1]), np.min(keypoint[..., 1])
            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                keypoint[..., 0] = (keypoint[..., 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
                keypoint[..., 1] = (keypoint[..., 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2
        else:
            h, w = results.get('img_shape', self.img_shape)
            keypoint[..., 0] = (keypoint[..., 0] - (w / 2)) / (w / 2)
            keypoint[..., 1] = (keypoint[..., 1] - (h / 2)) / (h / 2)

        if maskout is not None:
            keypoint[..., 0][maskout] = 0
            keypoint[..., 1][maskout] = 0
        results['keypoint'] = keypoint

        return results
    
class GenSkeFeat:
    def __call__(self, results):
        if 'keypoint_score' in results and 'keypoint' in results:
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        return results
        # return self.ops(results)

class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score" (optional), added or modified keys are
    "keypoint", "keypoint_score" (if applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    def __call__(self, results):

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'keypoint' in results:
            results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str

class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero', keypoint_str='keypoint'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode
        self.keypoint_str = keypoint_str

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results[self.keypoint_str]
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        nodes = results['closest_node'] if 'closest_node' in results else None
        oris = results['orientation'] if 'orientation' in results else None

        # M T V C
        origin_shape = keypoint.shape[0]
        if origin_shape < self.num_person:
            pad_dim = self.num_person - origin_shape
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and origin_shape == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

                if nodes is not None:
                    nodes = [nodes[0]] * self.num_person 

                if oris is not None:
                    oris = [oris[0]] * self.num_person 

        elif origin_shape > self.num_person:
            keypoint = keypoint[:self.num_person]
            if nodes is not None:
                nodes = nodes[:self.num_person]
            if oris is not None:
                oris = oris[:self.num_person]

        # TODO : Uncomment if num_clips usage
        if 'num_clips' in results:
            M, T, V, C = keypoint.shape
            nc = results.get('num_clips', 1)
            assert T % nc == 0
            keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4).reshape(nc * M, T // nc, V, C)
            keypoint = keypoint
        results[self.keypoint_str] = np.ascontiguousarray(keypoint)

        if nodes is not None:
            results['closest_node'] = nodes

        if oris is not None:
            results['orientation'] = np.array(oris, np.float32)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str
    
class FormatGCNInputMV(FormatGCNInput):
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero', num_view=3):
        super().__init__(num_person, mode)
        self.num_view = num_view

        # TODO : handle dimension where persons should be padded

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results[self.keypoint_str]

        # print("shape before format : ", results[self.keypoint_str].shape)

        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        n_in_person = keypoint.shape[0] // self.num_view

        # M T V C
        if n_in_person < self.num_person:
            keypoint = keypoint.reshape((self.num_view, n_in_person, ) + keypoint.shape[1:])
            # keypoint = keypoint.transpose(1, 0, 2, 3, 4)
            pad_dim = self.num_person - n_in_person
            # pad = np.zeros((pad_dim, ) + keypoint.shape[2:], dtype=keypoint.dtype)
            if self.mode == 'zero':
                keypoint = np.pad(
                    keypoint,
                    pad_width=((0, 0), (0, pad_dim), (0, 0), (0, 0), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            # elif self.mode == 'loop':
                # # pad = TODO
                # keypoint = np.concatenate([keypoint, pad], axis=1)
            # keypoint = keypoint.transpose(1, 0, 2, 3, 4)
            keypoint = keypoint.reshape(self.num_person*self.num_view, *keypoint.shape[2:])

            # print(keypoint[:, 0, 0, 0])
            # pad_dim = self.num_person - n_in_person
            # pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            # keypoint = np.insert(keypoint, list(range(1,(self.num_person * self.num_view) + 1, 1)), pad, axis=0) # TODO : check step. Here it considers each view having the same number of person and insert it each person after each view

        # TODO : check to collect only number of person per view
        elif keypoint.shape[0] > self.num_person*self.num_view:
            keypoint = keypoint[:self.num_person*self.num_view]

        # TODO : Uncomment if num_clips usage
        # M, T, V, C = keypoint.shape
        # nc = results.get('num_clips', 1)
        # assert T % nc == 0
        # keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)

        assert keypoint.shape[0] == self.num_person * self.num_view, str(keypoint.shape)
        results[self.keypoint_str] = np.ascontiguousarray(keypoint)

        # print("shape after format : ", results[self.keypoint_str].shape)


        return results

class Coco2H36m:
    def __call__(self, results):
        '''
            Input: x (M x T x V x C)
            
            COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
            
            H36M:
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        '''

        x = results['keypoint']

        y = np.zeros(x.shape)
        y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
        y[:,:,1,:] = x[:,:,12,:]
        y[:,:,2,:] = x[:,:,14,:]
        y[:,:,3,:] = x[:,:,16,:]
        y[:,:,4,:] = x[:,:,11,:]
        y[:,:,5,:] = x[:,:,13,:]
        y[:,:,6,:] = x[:,:,15,:]
        y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
        y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
        y[:,:,9,:] = x[:,:,0,:]
        y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
        y[:,:,11,:] = x[:,:,5,:]
        y[:,:,12,:] = x[:,:,7,:]
        y[:,:,13,:] = x[:,:,9,:]
        y[:,:,14,:] = x[:,:,6,:]
        y[:,:,15,:] = x[:,:,8,:]
        y[:,:,16,:] = x[:,:,10,:]

        results['keypoint'] = y

        return results
    
class PadTime:
    def __init__(self, max_length, pad_value=0.):
        self.max_length = max_length
        self.pad_value = pad_value

    def __call__(self, results):
        # TODO : concatenate with pad_value
        assert 'keypoint' in results, str(results.keys())
        keypoints = results['keypoint']
        M, T, V, C = keypoints.shape
        newkpts = np.zeros((M, self.max_length, V, C), dtype=keypoints.dtype)
        Tnz = T if T < self.max_length else self.max_length
        newkpts[:, :Tnz, :, :] = keypoints[:, :Tnz, :, :]
        results['keypoint'] = newkpts
        return results

class Centerize:
    def __init__(self, convention = 'coco'):
        self.joint_center_idx = -1
        if convention == 'coco':
            self.joint_center_idx = 20 #spine
        elif convention == 'nucla':
            self.joint_center_idx = 2 #
        elif convention == 'openpose':
            self.joint_center_idx = 1 #spine
        elif convention == 'nturgb+d':
            self.joint_center_idx = 0

    def __call__(self, results):

            keypoints = results['keypoint']

            # there's a freedom to choose the direction of local coordinate axes!
            # trajectory = keypoints[:, :, self.joint_center_idx]

            # let spine of each frame be the joint coordinate center
            keypoints = keypoints - keypoints[:, :, self.joint_center_idx:self.joint_center_idx+1]

            # works well with bone, but has negative effect with joint and distance gate
            # keypoints[:, :, self.joint_center_idx] = trajectory

            results['keypoint'] = keypoints

            return results

class Normalize:
    def __init__(self, num_joints):
        self.num_joints = num_joints

    def __call__(self, results):
        keypoints = results['keypoint']
        for ip, joints_person in enumerate(keypoints):
            scalerValue = joints_person
            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1
            scalerValue = np.reshape(scalerValue, (-1, 20, 3))
            keypoints[ip] = scalerValue
        results['keypoint'] = keypoints
        return results

class KeepIndexes:
    """ Keep given view indexes 
    * indexes: indices to keep
    * random: If a single random index should be kept
              (if len(indexes) == 1 => random from all
               if len(indexes) > 1 => random form indexes)
    * append_ori: append orientation given by selected index (from range -180:180:30)
    """
    def __init__(self, indexes=[0], random=False, append_ori=False, dim=0, n_randoms=1):
        self.indexes = indexes
        self.random = random
        self.append_ori = append_ori
        self.dim = dim
        self.n_randoms = n_randoms
    
    def __call__(self, results):
        assert "keypoint" in results, str(results.keys())

        indexes = self.indexes
        if self.random:
            if len(indexes) == 1:
                shape = results["keypoint"].shape
                indexes = random.sample(range(shape[self.dim]), 1)
                results['indexes'] = indexes
            elif self.n_randoms > 1:
                indexes = random.sample(self.indexes, self.n_randoms)
                results['indexes'] = indexes
            else:
                indexes = random.sample(self.indexes, 1)
                results['indexes'] = [self.indexes.index(i) for i in indexes] # TODO : check this
        else:
            results['indexes'] = indexes
        results['indexes'] = np.array(results['indexes'], dtype=np.int64)
        
        if self.dim == 0:
            results["keypoint"] = results["keypoint"][indexes]
        elif self.dim == 1:
            results["keypoint"] = results["keypoint"][:, indexes]
        elif self.dim == 2:
            results["keypoint"] = results["keypoint"][:, :, indexes]

        if self.append_ori:
            ran = list(range(-180,180,30))
            thetas = [ran[idx] for idx in indexes]
            thetas_rad = [math.radians(theta) for theta in thetas]
            thetas_cos_sin = [np.array((math.sin(t), math.cos(t)), np.float32) for t in thetas_rad]
            results['orientation'] = np.array(thetas_cos_sin, np.float32)

        return results

class SampleDynamicIndexes:
    def __init__(
        self,
        n_bins: int = 12,
        batch_index_size: int = 2,
        dim = 0,
        verbose=False,
        device="cpu"
    ):
        self.num_bins = n_bins
        self.batch_index_size = batch_index_size
        self.dim = dim
        self.verbose = verbose
        
        # difficulty memory (EMA)
        self.difficulty = torch.ones(self.num_bins, device=device)
        # self.last_indexes = self.sample_bins(self.batch_index_size)
        self.last_indexes = torch.tensor([0, n_bins // 2], device=device) # start with 2 views with max distance, will be updated at each epoch end

        if self.verbose:
            print(f"First indexes: {self.last_indexes.cpu().numpy()}")

    def __call__(self, results):
        assert "keypoint" in results, str(results.keys())

        indexes = self.last_indexes
        results['indexes'] = indexes.cpu().numpy()
        
        if self.dim == 0:
            results["keypoint"] = results["keypoint"][indexes]
        elif self.dim == 1:
            results["keypoint"] = results["keypoint"][:, indexes]
        elif self.dim == 2:
            results["keypoint"] = results["keypoint"][:, :, indexes]

        return results
            
    def sample_bins(
        self,
        batch_size: int,
        alpha: float = 0.6,
        eps: float = 0.1
    ):
        """
        Returns bin indices [B]
        """
        weights = (self.difficulty ** alpha) + eps
        probs = weights / weights.sum()

        return torch.multinomial(probs, batch_size, replacement=False)

    @torch.no_grad()
    def update_difficulty(
        self,
        bins,
        action_loss,
        view_conf=None,
        beta: float = 0.02,
        w_act: float = 1.0,
        w_view: float = 0.3,
    ):
        """
        bins: [B]
        action_loss: [B]
        view_conf: [B] or None
        ctr_dist: [B] or None
        """
        difficulty = w_act * action_loss

        if view_conf is not None:
            difficulty = difficulty + w_view * view_conf

        # aggregate per bin
        for b in bins.unique():
            mask = bins == b
            mean_diff = difficulty[mask].mean() # if ever not mean before
            self.difficulty[b] = (
                (1 - beta) * self.difficulty[b] + beta * mean_diff
            )

        # choose new indexes
        self.last_indexes = self.sample_bins(self.batch_index_size)
        if self.verbose:
            print(f"New indexes: {self.last_indexes.cpu().numpy()}")


    def update_with_losses(self, last_loss):
        # choose new indexes
        self.last_indexes = self.sample_bins(self.batch_index_size)
        if self.verbose:
            print(f"New indexes: {self.last_indexes.cpu().numpy()}")
        return

        self.update_difficulty(
            bins=self.last_indexes,
            action_loss=torch.tensor(last_loss["action"], device=self.difficulty.device),
            view_conf=torch.tensor(last_loss["view"], device=self.difficulty.device) if "view" in last_loss else None
        )
        if self.verbose:
            print(f"Updated difficulty: {self.difficulty.cpu().numpy()}")


    
class AppendToKeypoint:
    ''' Append the given key of the dict to the keypoint array '''
    def __init__(self, keyword):
        self.keyword = keyword

    def __call__(self, results):
        assert self.keyword in results, results.keys()
        assert 'keypoint' in results, results.keys()

        data = results[self.keyword]
        kpts = results['keypoint']
        assert len(data) == len(kpts)
        data = data.reshape(len(data), 1, 1, -1) # No happening by frame or joint for now
        data = np.broadcast_to(data, (len(data), kpts.shape[1], kpts.shape[2], data.shape[-1]))

        results['keypoint'] = np.concatenate([kpts, data], axis=-1)
        return results

class DuplicateKeypoints:
    def __init__(self, times=2, dim=0):
        self.times = times
        self.dim = dim

    def __call__(self, results):
        assert 'keypoint' in results, str(results.keys())
        keypoints = results['keypoint']
        keypoints = np.concatenate([keypoints]*self.times, axis=self.dim)
        results['keypoint'] = keypoints
        return results
    
class SubstituteKeypoint:
    def __init__(self, map: Dict[str, str]):
        self.map = map

    def __call__(self, results):
        keypoints = results['keypoint']

        for key, val in self.map.items():
            keypoints[:, :, key] = keypoints[:, :, val]

        results['keypoint'] = keypoints
        return results
    
class AddKeypoint:
    def __init__(self, map: Dict[str, str]):
        self.map = map

    def __call__(self, results):
        keypoints = results['keypoint']

        for key, val in self.map.items():
            keypoints[:, :, key] = keypoints[:, :, val]

        results['keypoint'] = keypoints
        return results