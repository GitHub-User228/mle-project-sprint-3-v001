import numpy as np
from typing import List

EARTH_AVERAGE_RADIUS = 6371


def spherical_to_cartesian(points: np.ndarray[float]) -> np.ndarray[float]:
    """
    Convert spherical coordinates (latitude, longitude) to Cartesian coordinates (x, y, z).

    Args:
        points (np.ndarray[float]):
            An array of spherical coordinates in the format [latitude, longitude].

    Returns:
        np.ndarray[float]:
            An array of Cartesian coordinates in the format [x, y, z].
    """
    points = np.radians(points)
    points2 = np.zeros((points.shape[0], 3))
    points2[:, 0] = np.cos(points[:, 0]) * np.cos(points[:, 1])
    points2[:, 1] = np.cos(points[:, 0]) * np.sin(points[:, 1])
    points2[:, 2] = np.sin(points[:, 0])
    return points2


def haversine_dist(
    lat1: List[float | int] | np.ndarray[float | int],
    lon1: List[float | int] | np.ndarray[float | int],
    lat2: List[float | int] | np.ndarray[float | int],
    lon2: List[float | int] | np.ndarray[float | int],
) -> np.ndarray[float]:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Args:
        lat1 (List[float | int] | np.ndarray[float | int]):
            The latitude of the first points.
        lon1 (List[float | int] | np.ndarray[float | int]):
            The longitude of the first points.
        lat2 (List[float | int] | np.ndarray[float | int]):
            The latitude of the second points.
        lon2 (List[float | int] | np.ndarray[float | int]):
            The longitude of the second points.

    Returns:
        np.ndarray[float]:
            The great circle distance between the two points in
            kilometers.
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return c * EARTH_AVERAGE_RADIUS


def haversine_dist2(
    points1: np.ndarray[float | int],
    points2: np.ndarray[float | int],
) -> np.ndarray[float]:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Args:
        points1 (np.ndarray[float | int]):
            The latitude and longitude of the first points.
        points2 (np.ndarray[float | int]):
            The latitude and longitude of the second points.

    Returns:
        np.ndarray[float]:
            The great circle distance between the two points in
            kilometers.
    """
    points1 = np.radians(points1)
    points2 = np.radians(points2)
    a = (
        np.sin((points2[:, 0] - points1[:, 0]) / 2) ** 2
        + np.cos(points1[:, 0])
        * np.cos(points2[:, 0])
        * np.sin((points2[:, 1] - points1[:, 1]) / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(a)) * EARTH_AVERAGE_RADIUS
