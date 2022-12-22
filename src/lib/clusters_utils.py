import numpy as np
from sklearn.neighbors import KDTree
import math
import seaborn as sns
import scipy as sc
from scipy.sparse import csr_matrix, lil_matrix
from collections import Counter
import functools
import operator

# Ellipsoid model constants (actual values here are for WGS84)
sm_a = 6378137.0
sm_b = 6356752.314
CLUSTER_SIZE = 255 * 30


def deg2rad(deg):
    return deg / 180.0 * 3.141


def rad2deg(rad):
    return rad / 3.141 * 180.0


def deg2planar(lats, lons):
    return list(map(lambda x, y: mercator(x, y), lats, lons))


# rad2deg, deg2rad, deg2planar, mercator, are taken from https://gis.stackexchange.com/questions/15269/how-to-convert-lat-long-to-meters-using-mercator-projection-in-c
def mercator(lat, lon):
    """
    LatLonToWorldMercator

     Converts a latitude/longitude pair to x and y coordinates in the
     World Mercator projection.

     Inputs:
       lat   - Latitude of the point.
       lon   - Longitude of the point.
       isDeg - Whether the given latitude and longitude are in degrees. If False
               (default) it is assumed they are in radians.

     Returns:
       x,y - A 2-element tuple with the World Mercator x and y values.

    """
    lon0 = 0
    lat = deg2rad(lat)
    lon = deg2rad(lon)
    x = sm_a * (lon - lon0)
    y = sm_a * math.log((math.sin(lat) + 1) / math.cos(lat))
    return x, y


def latLonToMeters(lat, lon):
    # This exploits information in satellite date gathering

    points = np.vstack((lat, lon)).T
    return np.array(deg2planar(points[:, 0], points[:, 1]))


def cluster_stats(points=None, lat=None, lon=None):
    """

    Args:
        points: Array of points in the form [[lat1, lon1], [lat2, lon2], ...]
        lat: Array of latitudes
        lon: Array of longitudes

    Returns: Some basic statistics on overlapping of the clusters centered on those points

    """
    if points is None:
        points = latLonToMeters(lat, lon)

    tree = KDTree(points, metric="manhattan")
    clusters_count = tree.query_radius(points, CLUSTER_SIZE, count_only=True)
    return {
        "min": np.min(clusters_count),
        "max": np.max(clusters_count),
        "mean": np.mean(clusters_count),
        "median": np.median(clusters_count),
        "std": np.std(clusters_count),
        # "probability overlapping": len(set().union(*clusters_count)) / len(points),
        "hist": sns.distplot(clusters_count, kde=False, bins=20),
    }


def construct_overlapping_graph(points):
    """

    Args:
        points: the centers of the clusters

    Returns: the adjacency matrix of the graph of overlapping clusters(an edge between overlapping clusters)

    """
    tree = KDTree(points, metric="manhattan")
    clusters = tree.query_radius(points, CLUSTER_SIZE)
    adj = lil_matrix((len(points), len(points)))
    for i, cluster in enumerate(clusters):
        adj[i, cluster] = 1

    return adj


def first_fit_strategy(components, k):
    """

    Args:
        components: An array mappping each sample to an indexed connected component
        k: the number of folds to fill

    Returns: an assignement of each sample to a fold trying to fill the folds as evenly as possible and such that every samples
    that are in the same connected component are in the same fold

    """
    integers = Counter(components)
    sorted_components = sorted(integers.items(), key=lambda item: item[1], reverse=True)
    final_dict = {}
    for i, p in enumerate(sorted_components):
        final_dict.setdefault(i % k, []).append(p[0])

    element_components = {}
    for i, p in enumerate(components):
        element_components.setdefault(p, []).append(i)

    final_dict = {k: [element_components.get(j) for j in v] for k, v in final_dict.items()}
    for k, v in final_dict.items():
        final_dict[k] = functools.reduce(operator.iconcat, v, [])

    return final_dict


def split_k_sets(k, lat=None, lon=None, points=None, strategy="first_fit"):
    """

    Args:
        k: Number of folds
        lat: Latitude of the centers of the clusters
        lon: Longitude of the centers of the clusters
        points: Array of points in the form [[lat1, lon1], [lat2, lon2], ...]
        strategy: For future work, the choice of the heuristic to use to split the data

    Returns: A dictionary mapping each fold to the indices of the samples in that fold

    """
    if points is None:
        points = latLonToMeters(lat, lon)
    adj = construct_overlapping_graph(points)
    n_components, components = sc.sparse.csgraph.connected_components(adj, directed=False, connection="weak",
                                                                      return_labels=True)
    if strategy == "first_fit":
        return first_fit_strategy(components, k)
    else:
        raise NotImplementedError


def plot_split_perf(split):
    """

    Args:
        split: a dictionary mapping each fold to the indices of the samples in that fold

    Returns: A plot of the distribution of the number of samples in each fold

    """
    sns.histplot(list(map(lambda x: len(x[1]), split.items())))


def folds_from_split(split_map):
    """

    Args:
        split_map: a dictionary mapping each fold to the indices of the samples in that fold

    Returns: a list of folds, each fold being a list of indices

    """
    for k, v in split_map.items():
        yield [i for (kk, vv) in split_map.items() for i in vv if kk != k], v


def save_lat_lon(lat, lon, path="../data/lat_lon.npz"):
    np.savez(path, lat=lat, lon=lon)


def load_lat_lon(path="../data/lat_lon.npz"):
    data = np.load(path)
    return data["lat"], data["lon"]
