# geometry.py

import numpy as np

def reflect_point_across_plane(point, plane):
    """
    Berechnet den Bildpunkt einer gegebenen Punktposition über eine Ebene.

    :param point: Ursprüngliche Punktposition als numpy-Array [x, y, z]
    :param plane: Ebene definiert durch [a, b, c, d] für ax + by + cz + d = 0
    :return: Bildpunkt als numpy-Array [x', y', z']
    """
    x_s, y_s, z_s = point
    a, b, c, d = plane
    denominator = a**2 + b**2 + c**2
    if denominator == 0:
        raise ValueError("Ungültige Ebene mit a^2 + b^2 + c^2 = 0.")
    factor = 2 * (a * x_s + b * y_s + c * z_s + d) / denominator
    x_prime = x_s - a * factor
    y_prime = y_s - b * factor
    z_prime = z_s - c * factor
    return np.array([x_prime, y_prime, z_prime])

def distance(point1, point2):
    """
    Berechnet die euklidische Entfernung zwischen zwei Punkten im 3D-Raum.

    :param point1: Erster Punkt als numpy-Array [x, y, z]
    :param point2: Zweiter Punkt als numpy-Array [x, y, z]
    :return: Entfernung als float
    """
    return np.linalg.norm(point1 - point2)
