import numpy as np
import math
import random

from .matrix import rotation_3d_x, rotation_3d_y, rotation_3d_z

def xy_randint(min_value, max_value, forbidden_range):
    min_f_range = forbidden_range[0]
    max_f_range = forbidden_range[1]
    while True:
        x = np.random.randint(min_value, max_value)
        y = np.random.randint(min_value, max_value)
        if x < min_f_range or y < min_f_range or x > max_f_range or y > max_f_range:
            return x, y

def random_camera_translation():
    x1000, y1000 = xy_randint(-6500, 7500, (-1000, 1000))
    x, y = x1000 / 1000., y1000 / 1000.
    z = np.random.randint(3000, 5000) / 1000.
    return np.array([x, y, z], dtype=float)

def compute_camera_rotation(translation):
    # pich angle
    x = translation[0]
    y = translation[1]
    z = translation[2]
    tanx = math.sqrt(x**2 + y**2) / z if z != 0 else math.tan(np.radians(90))
    x_angle = math.atan(tanx) * 180. / math.pi

    # roll angle
    y_angle = 0

    # yaw angle
    tanz = x / -y if abs(y) > 1e-3 else -math.inf
    z_angle = math.atan(tanz) * 180. / math.pi
    if y >= 0:
        z_angle += 180.

    return x_angle, y_angle, z_angle

def camera_pose(camera_translation, camera_angles, inverse=False, right2left=False):
    camera_rotation = rotation_3d_z(np.radians(camera_angles[2]))
    camera_rotation = camera_rotation @ rotation_3d_y(np.radians(camera_angles[1]))
    camera_rotation = camera_rotation @ rotation_3d_x(np.radians(camera_angles[0]))
    Rc = camera_rotation[:3, :3]

    if right2left:
        camera_rotation = camera_rotation @ rotation_3d_x(np.radians(180))
        camera_translation = [camera_translation[0], camera_translation[1], -camera_translation[2]]


    if not inverse:
        camera_pose = camera_rotation
        camera_pose[:3, 3] = camera_translation

    elif False: # estimation of inverse
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = Rc.transpose()

        Tc = np.asarray(camera_translation).transpose()
        trans_after_rot = -1. * Rc.transpose() @ Tc
        camera_pose[:3, 3] = trans_after_rot
    else:
        camera_pose = camera_rotation
        camera_pose[:3, 3] = camera_translation
        camera_pose = np.linalg.inv(camera_pose)  

    return camera_pose

def random_camera_extrinsic_params():
    rand_trans = random_camera_translation()
    cam_angles = compute_camera_rotation(rand_trans)
    return rand_trans, cam_angles

def random_camera_pose():
    rand_trans, cam_angles = random_camera_extrinsic_params()
    random_pose = camera_pose(rand_trans, cam_angles, inverse=False)
    return random_pose

def sampled_camera_poses(step_degree = 30):
    trans_z = 4
    radius = 7
    angles = range(-180, 180, step_degree)
    angles_rad = [math.radians(a) for a in angles]
    vertices = [[math.cos(a) * radius, math.sin(a) * radius, trans_z] for a in angles_rad]

    cams = [(t, compute_camera_rotation(t)) for t in vertices]
    return cams

def sampled_camera_poses_random(step_degree = 30):
    trans_z = np.random.randint(3000, 5000) / 1000.
    radius = np.random.randint(6500, 7500) / 1000.

    angles = range(-180, 180, step_degree)
    offset = random.sample(range(-step_degree//2, +step_degree//2), len(angles))
    angles = [a + o for a, o in zip(angles, offset)]


    angles_rad = [math.radians(a) for a in angles]
    vertices = [[math.cos(a) * radius, math.sin(a) * radius, trans_z] for a in angles_rad]

    cams = [(t, compute_camera_rotation(t)) for t in vertices]
    return cams

class Scene3D:
    def __init__(self, focal_length_mm=50, viewport_width=224, viewport_height=224):
        sensor_width = 36
        self.focal_length_mm = focal_length_mm
        self.focal_length = focal_length_mm / sensor_width * viewport_width
        self.camera_center = [viewport_width // 2, viewport_height // 2]

    def camera_pose(self, camera_translation, camera_angles, inverse=False, right2left=False):
        return camera_pose(camera_translation, camera_angles, inverse, right2left)

    def project_joints(self, joints, camera_translation, camera_angles):
        camera_pose = self.camera_pose([-camera_translation[0], camera_translation[1], camera_translation[2]], [camera_angles[0], -camera_angles[1], -camera_angles[2]], inverse=True)

        K = np.eye(3)
        K[0][0] = self.focal_length
        K[1][1] = self.focal_length
        K[0][2] = self.camera_center[0]
        K[1][2] = self.camera_center[1]

        Khomo = K @ np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
        P = Khomo @ camera_pose

        joints[:, 0] *= -1.

        jHomo = np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).transpose()
        joints2d = P @ jHomo
        joints2d[:, :] /= joints2d[2, :]
        return joints2d[:2, :].transpose()