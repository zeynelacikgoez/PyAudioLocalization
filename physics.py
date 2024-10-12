import numpy as np

def schallgeschwindigkeit(temp, feuchte, druck=101325):
    """
    Berechnet die Schallgeschwindigkeit in Luft unter Berücksichtigung von Temperatur, Luftfeuchtigkeit und Druck.
    Formel basierend auf der Formel von Cramer (1993):
    c = 331 + 0.6 * T + 0.0124 * RH + 0.0006 * (P / 1000)
    
    :param temp: Temperatur in °C
    :param feuchte: Luftfeuchtigkeit in %
    :param druck: Atmosphärischer Druck in Pa (Standard: 101325 Pa)
    :return: Schallgeschwindigkeit in m/s
    """
    # Temperatur in Kelvin (optional, falls benötigt für erweiterte Formeln)
    T_kelvin = temp + 273.15
    
    # Berechnung der Schallgeschwindigkeit
    c = 331 + 0.6 * temp + 0.0124 * feuchte + 0.0006 * (druck / 1000)
    
    return c


def distance(point1, point2):
    """
    Berechnet die euklidische Entfernung zwischen zwei Punkten im 3D-Raum.
    
    :param point1: Erster Punkt als numpy-Array [x, y, z]
    :param point2: Zweiter Punkt als numpy-Array [x, y, z]
    :return: Entfernung als float
    """
    return np.linalg.norm(point1 - point2)


def reflect_point_across_plane(point, plane):
    """
    Berechnet den Bildpunkt einer gegebenen Punktposition über eine Ebene.
    
    :param point: Ursprüngliche Punktposition als numpy-Array [x, y, z]
    :param plane: Ebene definiert durch [a, b, c, d] für ax + by + cz + d = 0
    :return: Bildpunkt als numpy-Array [x', y', z']
    :raises ValueError: Wenn die Ebenengleichung ungültig ist (a^2 + b^2 + c^2 = 0)
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
