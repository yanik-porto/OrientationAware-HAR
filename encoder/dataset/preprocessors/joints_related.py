import numpy as np
import random
import math
from abc import ABC, abstractmethod

from .scene_3d import Scene3D, random_camera_extrinsic_params, sampled_camera_poses, sampled_camera_poses_random, compute_camera_rotation
from .matrix import rotation_3d_z
from .geometry import get_unit_vector, create_realistic_mask


class Camera:
    def __init__(self, trans, rot):
        self.trans = trans
        self.rot = rot

def rotate_joints(rotmat, joints_to_rotate):
    jHomo = np.concatenate((joints_to_rotate, np.ones((joints_to_rotate.shape[0], 1))), axis=1).transpose()
    jRotated = rotmat @ jHomo
    joints_to_rotate = jRotated.transpose()[:, :3]
    return joints_to_rotate

def find_closest_angle(random_angle, angles_list):
    """
    Finds the closest angle from the predefined list to a given random angle.

    Args:
        random_angle: A float representing an angle in the range [-180, 150].

    Returns:
        A float representing the closest angle from the predefined list.
    """
    # predefined_angles = list(range(-180, 180, 30))
    normalized_angle =  (random_angle + 180) % 360 - 180

    # TODO : match -180 if closer to 180 than 150
    if normalized_angle > 165.:
        return -180

    closest_angle = angles_list[0]
    min_difference = abs(normalized_angle - closest_angle)

    for angle in angles_list[1:]:
        difference = abs(normalized_angle - angle)
        if difference < min_difference:
            min_difference = difference
            closest_angle = angle

    return closest_angle

class ProjectToCameras(ABC):
    """Rotate the joints with given angles and project in 2D to all cameras, defined in the child classes"""

    def __init__(self, width, height, random_angle=False, method='fixed', angle_degrees=[0],  convention='coco',
                 mask_self_occlusions=False, offset_to_canonical=False, save_orientation=False, save_cams=False):
        """method=['fixed', 'random_at_init', 'random_at_call']"""

        random.seed(255)

        self.width = width
        self.height = height
        self.scene = Scene3D(viewport_width=self.width, viewport_height=self.height)

        assert(method in ['fixed', 'random_at_init', 'random_at_call'])
        self.method = method

        self.random_angle = random_angle
        self.angle_degrees = angle_degrees

        self.keypoint_str = 'keypoint'

        if convention == 'coco':
            self.hips_idxs = (12, 11)
        elif convention == 'tsu':
            self.hips_idxs = (4, 3)
        elif convention == 'nturgb+d':
            self.hips_idxs = (16, 12)

        self.mask_self_occlusions = mask_self_occlusions
        self.offset_to_canonical = offset_to_canonical
        self.save_orientation = save_orientation
        self.save_cams = save_cams

    def create_rot_mat(self, first_frame, angle_degree):
        rotz = rotation_3d_z(np.radians(angle_degree))
        return rotz

    def project_to_camera(self, joints, cam_index, angle_degree):
        keypoints = []
        chosen_cam = self.cameras[cam_index]

        for joints_person in joints:
            first_frame = joints_person[0]
            rotmat = self.create_rot_mat(first_frame, angle_degree)
            if self.offset_to_canonical:
                rotmatoffset = self.create_rot_mat_offset(first_frame, cam_index)
            keypoints_person = []
            for joints_frame in joints_person:
                joints_frame = rotate_joints(rotmat, joints_frame)
                if self.mask_self_occlusions:
                    joints_frame = create_realistic_mask(joints_frame, chosen_cam.trans)
                    if self.offset_to_canonical:
                        joints_frame = rotate_joints(rotmatoffset, joints_frame)
                joints2d_frame = self.scene.project_joints(joints_frame, chosen_cam.trans, chosen_cam.rot)

                keypoints_person.append(joints2d_frame)
            keypoints.append(keypoints_person)

        return keypoints
    
    @abstractmethod
    def create_cameras(self, update_num_views=False, update_cam_indexes=True):
        pass
    
    def keep_existing(self, results, projections):
        if self.keypoint_str in results:
            existing_kpts = results[self.keypoint_str]
            assert(len(existing_kpts.shape) == 4)
            projections.append(existing_kpts)
        return projections

    def get_hips_angle(self, first_frame):
        lhip = first_frame[self.hips_idxs[1]]
        rhip = first_frame[self.hips_idxs[0]]
        unit_vec = get_unit_vector(rhip, lhip)
        angle = math.atan2(unit_vec[1], unit_vec[0])
        return angle

    def __call__(self, results):
        assert "joints" in results, str(results.keys())
        joints = results["joints"]

        if np.isnan(joints).any():
            print(np.isnan(joints).any())
            joints[np.isnan(joints)] = 0

        if len(joints.shape) == 3:
            joints = np.expand_dims(joints, axis=0)

        if self.random_angle:
            self.angle_degree = np.random.randint(0, 359)

        if self.method == 'random_at_call':
            self.create_cameras(False, True)

        projections = []
        projections = self.keep_existing(results, projections)

        oris = []
        for cam_index in self.cam_indexes:
            for angle_degree in self.angle_degrees:
                keypoints_projection = np.array(self.project_to_camera(joints, cam_index, angle_degree), dtype=np.float32)
                projections.append(keypoints_projection)

                if self.save_orientation:
                    for joints_person in joints:
                        first_frame = joints_person[0]
                        hips_angle = self.get_hips_angle(first_frame)
                        z_camera_angle = self.cameras[cam_index].rot[2]
                        ori = -hips_angle + np.radians(z_camera_angle) + np.radians(angle_degree)
                        oris.append(np.array([math.sin(ori), math.cos(ori)], dtype=np.float32))

        results[self.keypoint_str] = np.concatenate(projections) if len(projections) > 1 else projections[0]
        results['img_shape'] = (self.height, self.width)
        results['original_shape'] = (self.height, self.width)

        # hips_angles = []
        closest_node = []
        angles_list = list(range(-180, 180, 30))
        for joints_person in  joints:
            first_frame = joints_person[0]
            hips_angle = self.get_hips_angle(first_frame)
            closest = find_closest_angle(np.degrees(hips_angle), angles_list=angles_list)
            closest_node.append(angles_list.index(closest))
        results['closest_node'] = closest_node

        if self.save_orientation:
            results['orientation'] = np.array(oris, np.float32)

        if self.save_cams:
            cams = []
            for cam_index in self.cam_indexes:
                angles = self.cameras[cam_index].rot
                tr = self.cameras[cam_index].trans
                cam_coordinates  = [[tr[0], math.sin(np.radians(angles[0])), math.cos(np.radians(angles[0]))], [tr[1], math.sin(np.radians(angles[1])), math.cos(np.radians(angles[1]))], [tr[2], math.sin(np.radians(angles[2])), math.cos(np.radians(angles[2]))]]
                cams.append(np.array(cam_coordinates, dtype=np.float32))
            results["cameras"] = np.array(cams, dtype=np.float32)

        return results
    
    def create_rot_mat_offset(self, first_frame, cam_index):
        hips_angle = self.get_hips_angle(first_frame)
        z_camera_angle = self.cameras[cam_index].rot[2]
        closest = find_closest_angle(np.degrees(hips_angle), angles_list=list(range(-180, 180, 30)))

        # TODO : check if angle_degree has to be taken into account
        rotz_fix = rotation_3d_z(np.radians(z_camera_angle) + np.radians(closest))
        return rotz_fix

class ProjectToDefinedCams(ProjectToCameras):
    """Rotate the joints with given angles and project in 2D to defined cameras"""

    def __init__(self, width, height, random_angle=False, method='fixed', angle_degrees=[0], convention='coco', mask_self_occlusions=False, cam_indexes=[0], offset_to_canonical=False):
        """method=['fixed', 'random_at_init', 'random_at_call']"""
        super().__init__(width, height, random_angle, method, angle_degrees, convention, mask_self_occlusions, offset_to_canonical)
        self.cam_indexes = cam_indexes

        self.cameras = []
        self.cameras.append(Camera((7.35889, -6.92579, 4.95831), (63.5593, 0, 46.6919)))
        self.cameras.append(Camera((7.58162, 7.0136, 4.95831), (63.5593, 0, 130.047)))
        self.cameras.append(Camera((-6.40549, 7.0136, 4.95831), (63.5593, 0, 223.362)))
        self.create_cameras(update_cam_indexes = method=='random_at_init')

    def create_cameras(self, update_num_views=False, update_cam_indexes=True):
        # TODO : handle update_num_views
        self.cam_indexes = random.sample(range(len(self.cameras)), len(self.cam_indexes))  if update_cam_indexes else self.cam_indexes
        assert len(self.cam_indexes) > 0, "cam_indexes length : %d".format(len(self.cam_indexes))
        assert all(0 <= cam_index < len(self.cameras) for cam_index in self.cam_indexes), "Invalid cam_indexes : %s".format(self.cam_indexes)

    def update(self):
        if self.method == 'random_at_init':
            self.cam_indexes = random.sample(range(len(self.cameras)), len(self.cam_indexes))
        
class ProjectToRandomCamera(ProjectToCameras):
    """ Project 3D joints to a camera with random position, yaw and pitch. """

    def __init__(self, width, height, method='random_at_call', mask_self_occlusions=False, num_views=1, random_num_views=False, save_orientation=False, save_cams=False):
        """method=['random_at_init', 'random_at_call']"""
        super().__init__(width, height, False, method, [0], 'coco', mask_self_occlusions, save_orientation=save_orientation, save_cams=save_cams)

        self.num_views = num_views
        self.current_num_views = self.num_views
        self.random_num_views = random_num_views
        self.create_cameras(self.random_num_views, True)
    
    def create_cameras(self, update_num_views=False, update_cam_indexes=True):
        if update_num_views:
            self.current_num_views = np.random.randint(2, self.num_views)
        self.cameras = [Camera(*random_camera_extrinsic_params()) for _ in range(self.current_num_views)]
        if update_cam_indexes:
            self.cam_indexes = list(range(self.current_num_views))

    def update(self):
        if self.method == 'random_at_init':
            self.create_cameras(self.random_num_views)

class ProjectToGtCamera(ProjectToCameras):
    """ Rotate the sequence to be facing camera,
        by computing the rotation between the upper body
        of the first squeleton in the sequence and the camera. """

    def __init__(self, width, height, angle_degrees=[0],  convention='coco', mask_self_occlusions=False, save_orientation=False, camera_height=5.):
        super().__init__(width, height, False, 'fixed', angle_degrees, convention, mask_self_occlusions, save_orientation=save_orientation)

        self.cameras = []
        cam = Camera((0, -11, 5), (70, 0, 0))
        if camera_height != 5.:
            cam.trans = (cam.trans[0], cam.trans[1], camera_height)
            correct_rot = compute_camera_rotation(cam.trans)
            cam.rot = (correct_rot[0], cam.rot[1], cam.rot[2]) # correct the pitch according to the new height
        self.cameras.append(cam)
        self.create_cameras(False, True)

        self.keypoint_str = 'keypoint'

    def create_cameras(self, update_num_views=False, update_cam_indexes=True):
        self.cam_indexes = [0]

    def create_rot_mat(self, first_frame, angle_degree, cam_index=0):
        hips_angle = self.get_hips_angle(first_frame)
        z_camera_angle = self.cameras[cam_index].rot[2]
        rotz_fix = rotation_3d_z(-hips_angle + np.radians(z_camera_angle) + np.radians(angle_degree))
        return rotz_fix

class ProjectToClosestCamera(ProjectToCameras):
    """ Rotate the sequence to closest gt camera,
        by computing the rotation between the upper body
        of the first squeleton in the sequence and the camera. """

    def __init__(self, width, height, angle_degrees=[0],  convention='coco', mask_self_occlusions=False):
        super().__init__(width, height, False, 'fixed', angle_degrees, convention, mask_self_occlusions)

        self.cameras = []
        self.cameras.append(Camera((0, -11, 5), (70, 0, 0)))
        self.create_cameras(False, True)

        self.keypoint_str = 'keypoint'

    def create_cameras(self, update_num_views=False, update_cam_indexes=True):
        self.cam_indexes = [0]

    def create_rot_mat(self, first_frame, angle_degree, cam_index=0):
        hips_angle = self.get_hips_angle(first_frame)
        z_camera_angle = self.cameras[cam_index].rot[2]
        closest = find_closest_angle(np.degrees(hips_angle), angles_list=list(range(-180, 180, 30)))

        # TODO : check if angle_degree has to be taken into account
        rotz_fix = rotation_3d_z(-hips_angle + np.radians(z_camera_angle) + np.radians(closest))
        return rotz_fix

class ProjectToSampledCams(ProjectToDefinedCams):
    """Align the skeleton to z axis and project in 2D to defined cameras sampled around the world center """

    def __init__(self, width, height, random_angle=False, method='fixed', angle_degrees=[0], mask_self_occlusions=False, cam_indexes=[0], convention='coco'):
        """method=['fixed', 'random_at_init', 'random_at_call']"""
        super().__init__(width, height, random_angle, method, angle_degrees, convention, mask_self_occlusions, cam_indexes)
        self.cam_indexes = cam_indexes

        self.cameras = [Camera(cam[0], cam[1]) for cam in sampled_camera_poses(step_degree = 30)]
        self.create_cameras(update_cam_indexes = method=='random_at_init')

    def create_rot_mat(self, first_frame, angle_degree, cam_index=0):
        hips_angle = self.get_hips_angle(first_frame)
        rotz_fix = rotation_3d_z(-hips_angle + np.radians(angle_degree))
        return rotz_fix
    
class ProjectToRandomSampledCams(ProjectToDefinedCams):
    """Align the skeleton to z axis and project in 2D to defined cameras sampled around the world center """

    def __init__(self, width, height, random_angle=False, method='fixed', angle_degrees=[0], mask_self_occlusions=False, cam_indexes=[0], convention='coco', offset_to_canonical=True):
        """method=['fixed', 'random_at_init', 'random_at_call']"""
        super().__init__(width, height, random_angle, method, angle_degrees, convention, mask_self_occlusions, cam_indexes, offset_to_canonical)
        self.cam_indexes = cam_indexes

        # self.cameras = [Camera(cam[0], cam[1]) for cam in sampled_camera_poses(step_degree = 30)]
        self.create_cameras(update_cam_indexes = method=='random_at_init')

    def create_rot_mat(self, first_frame, angle_degree, cam_index=0):
        hips_angle = self.get_hips_angle(first_frame)
        rotz_fix = rotation_3d_z(-hips_angle + np.radians(angle_degree))
        return rotz_fix
    
    def create_cameras(self, update_num_views=False, update_cam_indexes=True):
        self.cameras = [Camera(cam[0], cam[1]) for cam in sampled_camera_poses_random(step_degree = 30)]
    
    def update(self):
        if self.method == 'random_at_init':
            self.create_cameras()

class JointsToKeypoints:
    """ Just move the joint input inside the keypoint input"""

    def __init__(self, convention='coco', fix_hips=False, random_angle=False):
        if convention == 'coco':
            self.hips_idxs = (12, 11) # TODO : check but seems that -1 is needed
        elif convention == 'tsu':
            self.hips_idxs = (4, 3) # TODO : check this, res totally different than original
        elif convention == 'nturgb+d':
            self.hips_idxs = (16, 12)

        self.fix_hips = fix_hips
        self.random_angle = random_angle

    def get_hips_angle(self, first_frame):
        lhip = first_frame[self.hips_idxs[1]]
        rhip = first_frame[self.hips_idxs[0]]
        unit_vec = get_unit_vector(rhip, lhip)
        angle = math.atan2(unit_vec[1], unit_vec[0])
        return angle

    def create_rot_mat(self, first_frame, angle_degree, cam_index=0):
        hips_angle = self.get_hips_angle(first_frame)
        rotz_fix = rotation_3d_z(-hips_angle + np.radians(angle_degree))
        return rotz_fix

    def rand_view_transform(self,X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def rotate_randomly(self, value):
        agx = np.random.randint(-60, 60)
        agy = np.random.randint(-60, 60)
        s = np.random.uniform(0.5, 1.5)

        center = value[0,1,:]
        value = value - center
        scalerValue = self.rand_view_transform(value, agx, agy, s)
        return scalerValue
        
    def __call__(self, results):
        assert "joints" in results, str(results.keys())

        joints = results['joints']

        if self.fix_hips:
            for ip, joints_person in enumerate(joints):
                rotmat = self.create_rot_mat(joints_person[0], 0)

                for iF in range(len(joints_person)):
                    joints_person[iF] = rotate_joints(rotmat, joints_person[iF])

                joints[ip] = joints_person

        if self.random_angle:
            for ip, joints_person in enumerate(joints):
                joints_person = self.rotate_randomly(joints_person)
                joints[ip] = joints_person

        results['keypoint'] = joints
        return results
    
class InverseAxis:
    """ Swap x y z axes"""
    def __init__(self, new_axes=[0, 2, 1]):
        self.new_axes = new_axes

    def __call__(self, results):
        assert "joints" in results, str(results.keys())


        joints = results['joints']
        joints[..., 0], joints[..., 1], joints[..., 2] = joints[..., self.new_axes[0]].copy(), joints[..., self.new_axes[1]].copy(), joints[..., self.new_axes[2]].copy()
        results['joints'] = joints
        return results

class CenterizeJoints:
    def __init__(self, convention = 'coco', first_frame_only=False):
        self.joint_center_idx = -1
        if convention == 'coco':
            self.joint_center_idx = 0 #spine
            # self.joint_center_idx = 20 #spine
        elif convention == 'nucla':
            self.joint_center_idx = 2 #
        elif convention == 'openpose':
            self.joint_center_idx = 1 #spine
        elif convention == 'tsu':
            self.joint_center_idx = 13
        elif convention == 'nturgb+d':
            self.joint_center_idx = 0

        self.first_frame_only = first_frame_only
        
    def __call__(self, results):

            joints = results['joints']

            # there's a freedom to choose the direction of local coordinate axes!
            # trajectory = keypoints[:, :, self.joint_center_idx]
            # print(self.joint_center_idx)

            # let spine of each frame be the joint coordinate center
            if self.first_frame_only:
                joints = joints - joints[:, 0:1, self.joint_center_idx:self.joint_center_idx+1]
            else:
                joints = joints - joints[:, :, self.joint_center_idx:self.joint_center_idx+1]

            # works well with bone, but has negative effect with joint and distance gate
            # keypoints[:, :, self.joint_center_idx] = trajectory

            results['joints'] = joints

            return results

class PreNormalize3D:
    """PreNormalize for NTURGB+D 3D keypoints (x, y, z)."""

    def unit_vector(self, vector):
        """Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'. """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis, theta):
        """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def __init__(self, zaxis=[0, 1], xaxis=[8, 4], align_spine=True, align_center=True):
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_spine = align_spine
        self.align_center = align_center

    def __call__(self, results):
        skeleton = results['joints']
        total_frames = results.get('total_frames', skeleton.shape[1])

        M, T, V, C = skeleton.shape
        assert T == total_frames
        if skeleton.sum() == 0:
            return results

        index0 = [i for i in range(T) if not np.all(np.isclose(skeleton[0, i], 0))]

        assert M in [1, 2]
        if M == 2:
            index1 = [i for i in range(T) if not np.all(np.isclose(skeleton[1, i], 0))]
            if len(index0) < len(index1):
                skeleton = skeleton[:, np.array(index1)]
                skeleton = skeleton[[1, 0]]
            else:
                skeleton = skeleton[:, np.array(index0)]
        else:
            skeleton = skeleton[:, np.array(index0)]

        T_new = skeleton.shape[1]

        if self.align_center:
            if skeleton.shape[2] == 25:
                main_body_center = skeleton[0, 0, 1].copy()
            else:
                main_body_center = skeleton[0, 0, -1].copy()
            mask = ((skeleton != 0).sum(-1) > 0)[..., None]
            skeleton = (skeleton - main_body_center) * mask

        if self.align_spine:
            joint_bottom = skeleton[0, 0, self.zaxis[0]]
            joint_top = skeleton[0, 0, self.zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)

            joint_rshoulder = skeleton[0, 0, self.xaxis[0]]
            joint_lshoulder = skeleton[0, 0, self.xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)

        results['joints'] = skeleton
        results['total_frames'] = T_new
        results['body_center'] = main_body_center
        return results

class RandomRot:

    def __init__(self, theta=0.3):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, results):
        skeleton = results['joints']
        M, T, V, C = skeleton.shape

        if np.all(np.isclose(skeleton, 0)):
            return results

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=3)
            rot_mat = self._rot3d(theta)
        elif C == 2:
            theta = np.random.uniform(-self.theta)
            rot_mat = self._rot2d(theta)
        results['joints'] = np.einsum('ab,mtvb->mtva', rot_mat, skeleton)

        return results

class RandomRotOneAxis:

    def __init__(self, theta=0.3, axis=1, append_index=False):
        self.theta = theta
        self.axis = axis
        self.append_index = append_index

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, results):
        skeleton = results['joints']
        M, T, V, C = skeleton.shape

        if np.all(np.isclose(skeleton, 0)):
            return results

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=1)
            theta_vec = np.array([0., 0., 0.])
            theta_vec[self.axis] = theta[0]
            rot_mat = self._rot3d(theta_vec)

        elif C == 2:
            theta = np.random.uniform(-self.theta)
            rot_mat = self._rot2d(theta)
            # TODO : check if still usefull to have C == 2 in this class

        results['joints'] = np.einsum('ab,mtvb->mtva', rot_mat, skeleton)

        if self.append_index:
            angles_list = list(range(-180, 180, 30))
            closest_angle = find_closest_angle(np.degrees(theta[0]), angles_list)
            results['indexes'] = np.array([angles_list.index(closest_angle)], dtype=np.int64)

        return results