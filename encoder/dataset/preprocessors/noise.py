    
import torch
import pickle
import os.path as op
import numpy as np

class AddNoiseToKeypoints:
    def __init__(self, doMasking=False, addScores=False, concatenate=False, reproducible=False, popLastAsGt=False, onlyFirstAsGt=True):

        mask_ratio = 0.05
        mask_T_ratio = 0.1
        noise_path = op.join(op.dirname(__file__), "params/synthetic_noise.pth")
        d2c_params_path = op.join(op.dirname(__file__), "params/d2c_params.pkl")
        self.aug = Augmenter2D(d2c_params_path, noise_path, mask_ratio, mask_T_ratio)
        self.doMasking = doMasking
        self.addScores = addScores
        self.concat = concatenate
        self.reproducible = reproducible
        self.popLastAsGt = popLastAsGt
        self.onlyFirstAsGt = onlyFirstAsGt
        self.seed = 255

    def __call__(self, results):
        assert 'keypoint' in results, str(results.keys())
        keypoints = results['keypoint']

        if 'keypoint_gt' not in results:
            if self.popLastAsGt:
                results['keypoint_gt'], keypoints = np.expand_dims(keypoints[-1], axis=0), keypoints[:-1]
            elif self.onlyFirstAsGt:
                results['keypoint_gt'] =  np.expand_dims(np.copy(keypoints[0]), axis=0)
            else:
                results['keypoint_gt'] = np.copy(keypoints)

        kpts_torch = torch.from_numpy(keypoints)
        kshape = kpts_torch.shape 
        if len(kshape) > 4:
            kpts_torch = torch.reshape(kpts_torch, (-1, *kshape[-3:]))

        if self.reproducible:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        kpts_torch, kpts_scores_torch = self.aug.add_noise(kpts_torch)
        if self.doMasking:
            kpts = self.aug.add_mask(torch.cat((kpts_torch, kpts_scores_torch), dim=-1))
            kpts_torch = kpts[..., :2]
            kpts_scores_torch = kpts[..., 2]

        kpts_torch = torch.reshape(kpts_torch, kshape)
        kpts_scores_torch = torch.reshape(kpts_scores_torch, kshape[:-1])

        results['keypoint'] = kpts_torch.cpu().detach().numpy()
        kpts_scores = kpts_scores_torch.cpu().detach().numpy()
        if self.addScores:
            results['keypoint_score'] = kpts_scores
        
        if self.concat:
            results['keypoint'] = np.concatenate((results['keypoint'], np.expand_dims(kpts_scores, axis=-1)), axis=-1)

        return results

class Augmenter2D(object):
    """
        Make 2D augmentations on the fly. PyTorch batch-processing GPU version.
    """
    def __init__(self, d2c_params_path, noise_path, mask_ratio, mask_T_ratio):
        with open(d2c_params_path, 'rb') as f: self.d2c_params = pickle.load(f)
        self.noise = torch.load(noise_path) # TODO : change for parameterics number of joints
        self.mask_ratio = mask_ratio
        self.mask_T_ratio = mask_T_ratio
        self.num_Kframes = 27
        self.noise_std = 0.002

    def dis2conf(self, dis, a, b, m, s):
        f = a/(dis+a)+b*dis
        shift = torch.randn(*dis.shape)*s + m
        # if torch.cuda.is_available():
        shift = shift.to(dis.device)
        return f + shift
    
    def add_noise(self, motion_2d):
        a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
        if "uniform_range" in self.noise.keys():
            uniform_range = self.noise["uniform_range"]
        else:
            uniform_range = 0.06
        motion_2d = motion_2d[:,:,:,:2]
        batch_size = motion_2d.shape[0]
        num_frames = motion_2d.shape[1]
        num_joints = motion_2d.shape[2]
        mean = self.noise['mean'].float()
        std = self.noise['std'].float()
        weight = self.noise['weight'][:,None].float()
        sel = torch.rand((batch_size, self.num_Kframes, num_joints, 1))
        gaussian_sample = (torch.randn(batch_size, self.num_Kframes, num_joints, 2) * std + mean)
        uniform_sample = (torch.rand((batch_size, self.num_Kframes, num_joints, 2))-0.5) * uniform_range
        noise_mean = 0
        delta_noise = torch.randn(num_frames, num_joints, 2) * self.noise_std + noise_mean
        # if torch.cuda.is_available():
        mean = mean.to(motion_2d.device)
        std = std.to(motion_2d.device)
        weight = weight.to(motion_2d.device)
        gaussian_sample = gaussian_sample.to(motion_2d.device)
        uniform_sample = uniform_sample.to(motion_2d.device)
        sel = sel.to(motion_2d.device)
        delta_noise = delta_noise.to(motion_2d.device)
            
        delta = gaussian_sample*(sel<weight) + uniform_sample*(sel>=weight)
        delta_expand = torch.nn.functional.interpolate(delta.unsqueeze(1), [num_frames, num_joints, 2], mode='trilinear', align_corners=True)[:,0]
        delta_final = delta_expand + delta_noise      
        motion_2d = motion_2d + delta_final
        dx = delta_final[:,:,:,0]
        dy = delta_final[:,:,:,1]
        dis2 = dx*dx+dy*dy
        dis = torch.sqrt(dis2)
        conf = self.dis2conf(dis, a, b, m, s).clip(0,1).reshape([batch_size, num_frames, num_joints, -1])
        return motion_2d, conf
        # return torch.cat((motion_2d, conf), dim=3)
        
    def add_mask(self, x):
        ''' motion_2d: (N,T,17,3)
        '''
        N,T,J,C = x.shape
        mask = torch.rand(N,T,J,1, dtype=x.dtype, device=x.device) > self.mask_ratio
        # x = x * mask
        # mask_T = torch.rand(1,T,1,1, dtype=x.dtype, device=x.device) > self.mask_T_ratio
        mask_T = torch.rand(N,T,1,1, dtype=x.dtype, device=x.device) > self.mask_T_ratio
        x = x * mask * mask_T
        return x

    def augment2D(self, motion_2d, mask=False, noise=False):
        if noise:
            motion_2d = self.add_noise(motion_2d)
        if mask:
            motion_2d = self.add_mask(motion_2d)
        return motion_2d
    
class MaskViews:
    def __init__(self):
        pass

    def __call__(self, results):
        assert 'keypoint' in results, str(results.keys())
        keypoints = results['keypoint']
        sz = keypoints.shape

        mask_all = np.zeros(sz, dtype=bool)
        # mask_view = np.ones((1, *sz[1:]), dtype=bool)
        mask_all[0, ...] = True
        # mask_all[0:12:2, ...] = True

        keypoints = keypoints * mask_all
        results['keypoint'] = keypoints
        return results

class MaskViewsRandom:
    def __init__(self, reproducible=False, mask_ratio = 0.4):
        self.seed = 255
        self.mask_ratio = mask_ratio
        self.reproducible = reproducible

    def __call__(self, results):

        if self.reproducible:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        assert 'keypoint' in results, str(results.keys())
        keypoints = results['keypoint']

        mask_all = np.random.rand(len(keypoints), 1, 1, 1) > self.mask_ratio

        keypoints = keypoints * mask_all
        results['keypoint'] = keypoints
        return results
