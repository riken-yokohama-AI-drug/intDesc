"""computation process definition module"""
import math

import numpy as np


def distance_two_points(point_a, point_b):
    """Distance between two points

    Args:
        point_a (list): 3D coordinates of point A
        point_b (list): 3D coordinates of point B

    Returns:
        float: Distance between two points
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    distance = np.linalg.norm(point_a - point_b)
    return distance


def vector_two_points(point_a, point_b):
    """vector formed by two points

    Args:
        point_a (list): 3D coordinates of point A
        point_b (list): 3D coordinates of point B

    Returns:
        list: vector formed by two points
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    vector = point_b - point_a
    return vector.tolist()


def angle_two_vectors(vector_a, vector_b):
    """Angle between two vectors

    Args:
        vector_a (list): Components of vector A
        vector_b (list): Components of vector B

    Returns:
        float: Angle between two vectors [degrees].
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    dot = np.dot(vector_a, vector_b)
    magnitude = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    rad = np.arccos(dot / magnitude)
    deg = math.degrees(rad)
    return deg


def angle_three_points(point_a, point_b, point_c):
    """angle of intersection of three points

    Args:
        point_a (list): 3D coordinates of point A
        point_b (list): 3D coordinates of point B
        point_c (list): 3D coordinates of point C

    Returns:
        float: Angle between 3 points [degrees].
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    return angle_two_vectors(vector_ba, vector_bc)


def normal_vector_three_points(point_a, point_b, point_c):
    """Normal vector formed by 3 points

    Args:
        point_a (list): 3D coordinates of point A
        point_b (list): 3D coordinates of point B
        point_c (list): 3D coordinates of point C

    Returns:
        list: Normal vector formed by 3 points
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b
    cross = np.cross(vector_ba, vector_bc)
    return cross.tolist()


def distance_point_to_plane(point_a, point_b, point_c, point_p):
    """Distance between a point and a plane

    Args:
        point_a (list): 3D coordinates of point A (a point constituting a plane)
        point_b (list): 3D coordinates of point B (a point constituting a plane)
        point_c (list): 3D coordinates of point C (a point constituting a plane)
        point_p (list): 3D coordinates of point P (point to measure distance to plane)

    Returns:
        float: Distance between a point and a plane
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
    point_p = np.array(point_p)

    normal_vector = normal_vector_three_points(point_a, point_b, point_c)
    unit_vector = normal_vector / np.linalg.norm(normal_vector)
    distance = np.linalg.norm(np.dot(unit_vector, point_p) - np.dot(unit_vector, point_a))

    return distance


def intersection_point_vertical_line_and_plane(point_a, point_b, point_c, point_p):
    """Intersection of a perpendicular line down from a point and a plane

    Args:
        point_a (list): 3D coordinates of point A (a point constituting a plane)
        point_b (list): 3D coordinates of point B (a point constituting a plane)
        point_c (list): 3D coordinates of point C (a point constituting a plane)
        point_p (list): 3D coordinates of point P (the point perpendicular to the plane)

    Returns:
        list: Intersection of a perpendicular line down from a point and a plane
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
    point_p = np.array(point_p)

    normal_vector = normal_vector_three_points(point_a, point_b, point_c)
    unit_vector = normal_vector / np.linalg.norm(normal_vector)

    vector_pa = point_a - point_p
    vector_oh = point_p + np.dot(unit_vector, vector_pa) * unit_vector

    return vector_oh.tolist()


def intersection_point_vertical_line_and_line(vector_v, point_a, point_p):
    """Intersection of a perpendicular line drawn from a point and a line defined by a vector

    Args:
        vector_v (list): Components of vector v
        point_a (list): 3D coordinates of point A that lies on the line formed by vector v
        point_p (list): 3D coordinates of point P

    Returns:
        list: Intersection of a perpendicular line down from a point and a plane
    """
    vector_v = np.array(vector_v)
    assert np.linalg.norm(vector_v) != 0.0, "Vector v magnitude is 0"
    point_a = np.array(point_a)
    point_p = np.array(point_p)

    vector_ap = point_p - point_a
    unit_vector = vector_v / np.linalg.norm(vector_v)
    vector_ab = np.dot(unit_vector, vector_ap) * unit_vector
    vector_ob = point_a + vector_ab

    return vector_ob.tolist()


def dihedral_angle_four_points(point_a, point_b, point_c, point_d):
    """Dihedral angle formed by four points

    Args:
        point_a (list): 3D coordinates of point A
        point_b (list): 3D coordinates of point B
        point_c (list): 3D coordinates of point C
        point_d (list): 3D coordinates of point D

    Returns:
        float: Dihedral angle formed by 4 points [degrees].
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
    point_d = np.array(point_d)

    normal_vector_b = normal_vector_three_points(point_a, point_b, point_c)
    normal_vector_c = normal_vector_three_points(point_b, point_c, point_d)

    return angle_two_vectors(normal_vector_b, normal_vector_c)


def center_of_gravity(points):
    """Center of mass between N points

    Args:
        points (list): Two-dimensional array with "N x 3" number of elements

    Returns:
        list: Center of mass between N points
    """
    points = np.array(points, dtype=object)
    mean = np.mean(points, axis=0, keepdims=True)[0]
    return mean.tolist()
