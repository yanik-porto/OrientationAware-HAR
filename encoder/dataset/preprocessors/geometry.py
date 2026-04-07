import math
import numpy as np

def get_unit_vector(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    magnitude = math.sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return (0, 0)  # Handle zero-length vector

    return dx / magnitude, dy / magnitude

def vector_mesh_intersection(vector_start, vector_end, mesh_vertices, mesh_faces):
    """https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    Checks if a vector defined by a start and end point intersects with a 3D mesh.
    
    Parameters:
    vector_start (numpy.ndarray): 3D starting point of the vector
    vector_end (numpy.ndarray): 3D ending point of the vector
    mesh_vertices (numpy.ndarray): 3D coordinates of the mesh vertices
    mesh_faces (numpy.ndarray): Indices of the vertices that form each triangle face
    
    Returns:
    bool: True if the vector intersects the mesh, False otherwise"""

    # Calculate the vector direction
    vector_direction = vector_end - vector_start
    
    # Loop through each triangle face in the mesh
    for face in mesh_faces:
        v0 = mesh_vertices[face[0]]
        v1 = mesh_vertices[face[1]]
        v2 = mesh_vertices[face[2]]

        # Calculate edges and normal of the triangle
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)

        # Calculate the determinant
        det = np.dot(-vector_direction, normal)

        # If the vector is parallel to the triangle, there is no intersection
        if np.abs(det) < 1e-8:
            continue

        inv_det = 1.0 / det
        p = vector_start - v0

        # Calculate the distance to the intersection point
        t = np.dot(normal, p) * inv_det
        if t < 0.0 or t > 1.0:
            continue
        
        u0 = np.cross(e2, -vector_direction)
        u = np.dot(u0, p) * inv_det
        v0 = np.cross(-vector_direction, e1)
        v = np.dot(v0, p) * inv_det

        # check if the interesection point is in the parallelogram formed by v0 and vectors v01 and v02
        if u < 0.0 or u > 1.0 or v < 0.0 or u + v > 1.0:
            continue

        # check if intersection point is outside the triangle 
        if u + v < 0.0 or u + v > 1.0:
            continue

        return True

    return False


def create_realistic_mask(vertices, cam_trans):
    rshould = 6
    lshould = 5
    rhip = 12
    lhip = 11
    body_joints = (6, 5, 12, 11)

    faces = ((rshould, lshould, lhip), (lhip, rhip, rshould))

    mask = np.ones(vertices.shape, dtype=vertices.dtype)
    for j in range(len(vertices)):
        if j in body_joints:
            continue
        intersect = vector_mesh_intersection(cam_trans, vertices[j], vertices, faces)
        # if intersect:
            # print(f"{j}th joint is occluded")
        mask[j] = float(not intersect)

    vertices = vertices * mask

    return vertices