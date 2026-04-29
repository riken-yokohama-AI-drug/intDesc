"""Mathematical utility functions for geometric calculations."""
import math

import numpy as np


def distance_two_points(point_a, point_b):
    """Distance between two points.

    Args:
        point_a (list): 3D coordinates of point A.
        point_b (list): 3D coordinates of point B.

    Returns:
        float: Distance between the two points.
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    distance = np.linalg.norm(point_a - point_b)
    return distance


def vector_two_points(point_a, point_b):
    """Vector defined by two points.

    Args:
        point_a (list): 3D coordinates of point A.
        point_b (list): 3D coordinates of point B.

    Returns:
        list: Vector from point A to point B.
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    vector = point_b - point_a
    return vector.tolist()


def angle_two_vectors(vector_a, vector_b):
    """Angle between two vectors.

    Args:
        vector_a (list): Components of vector A.
        vector_b (list): Components of vector B.

    Returns:
        float: Angle between the two vectors [degrees].
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    dot = np.dot(vector_a, vector_b)
    magnitude = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    rad = np.arccos(dot / magnitude)
    deg = math.degrees(rad)
    return deg


def angle_three_points(point_a, point_b, point_c):
    """Angle formed by three points.

    Args:
        point_a (list): 3D coordinates of point A.
        point_b (list): 3D coordinates of point B.
        point_c (list): 3D coordinates of point C.

    Returns:
        float: Angle formed by the three points [degrees].
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    return angle_two_vectors(vector_ba, vector_bc)


def normal_vector_three_points(point_a, point_b, point_c):
    """Normal vector defined by three points.

    Args:
        point_a (list): 3D coordinates of point A.
        point_b (list): 3D coordinates of point B.
        point_c (list): 3D coordinates of point C.

    Returns:
        list: Normal vector defined by the three points.
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b
    cross = np.cross(vector_ba, vector_bc)
    return cross.tolist()


def distance_point_to_plane(point_a, point_b, point_c, point_p):
    """Distance between a point and a plane.

    Args:
        point_a (list): 3D coordinates of point A defining the plane.
        point_b (list): 3D coordinates of point B defining the plane.
        point_c (list): 3D coordinates of point C defining the plane.
        point_p (list): 3D coordinates of point P whose distance to the plane is measured.

    Returns:
        float: Distance between the point and the plane.
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
    """Intersection between a perpendicular from a point and a plane.

    Args:
        point_a (list): 3D coordinates of point A defining the plane.
        point_b (list): 3D coordinates of point B defining the plane.
        point_c (list): 3D coordinates of point C defining the plane.
        point_p (list): 3D coordinates of point P from which the perpendicular is dropped.

    Returns:
        list: Intersection point between the perpendicular and the line.
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
    """Intersection between a perpendicular from a point and a line defined by a vector.

    Args:
        vector_v (list): Components of vector v.
        point_a (list): 3D coordinates of point A on the line defined by vector v.
        point_p (list): 3D coordinates of point P.

    Returns:
        list: Intersection point between the perpendicular and the line.
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
    """Dihedral angle defined by four points.

    Args:
        point_a (list): 3D coordinates of point A.
        point_b (list): 3D coordinates of point B.
        point_c (list): 3D coordinates of point C.
        point_d (list): 3D coordinates of point D.

    Returns:
        float: Dihedral angle defined by the four points [degrees].
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
    point_d = np.array(point_d)

    normal_vector_b = normal_vector_three_points(point_a, point_b, point_c)
    normal_vector_c = normal_vector_three_points(point_b, point_c, point_d)

    return angle_two_vectors(normal_vector_b, normal_vector_c)


def center_of_gravity(points):
    """Center of gravity of N points.

    Args:
        points (list): Two-dimensional array with shape N x 3.

    Returns:
        list: Center of gravity of the N points.
    """
    points = np.array(points, dtype=object)
    mean = np.mean(points, axis=0, keepdims=True)[0]
    return mean.tolist()
