import math
import numpy as np

def rotation_3d_x(alpha):
    R = np.eye(4)
    R[1, 1] = math.cos(alpha)
    R[1, 2] = -math.sin(alpha)
    R[2, 1] = math.sin(alpha)
    R[2, 2] = math.cos(alpha)
    return R

def rotation_3d_y(alpha):
    R = np.eye(4)
    R[0, 0] = math.cos(alpha)
    R[0, 2] = math.sin(alpha)
    R[2, 0] = -math.sin(alpha)
    R[2, 2] = math.cos(alpha)
    return R

def rotation_3d_z(alpha):
    R = np.eye(4)
    R[0, 0] = math.cos(alpha)
    R[0, 1] = -math.sin(alpha)
    R[1, 0] = math.sin(alpha)
    R[1, 1] = math.cos(alpha)
    return R