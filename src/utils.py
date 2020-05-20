import numpy as np

def calculate_centroid(data):
    return np.mean(data, axis=0)


def find_dimension_with_greatest_spread(data):
    return np.argmax(np.max(data, axis=0) - np.min(data, axis=0))


def calculate_radii(centroid, data):
    diff = np.tile(centroid, (data.shape[0], 1)) - data  # TODO make more efficient
    radii = np.linalg.norm(diff, axis=1)
    return radii 


def calculate_max_radius(centroid, data):
    radii = calculate_radii(centroid, data)
    max_radius_index = np.argmax(radii)
    max_radius = radii[max_radius_index]
    return max_radius


def calculate_min_radius(centroid, data):
    radii = calculate_radii(centroid, data)
    min_radius_index = np.argmin(radii)
    min_radius = radii[min_radius_index]
    return data[min_radius_index], min_radius

def brute_force_nn(q, data):
    diff = np.tile(q, (data.shape[0], 1)) - data
    radii = np.linalg.norm(diff, axis=1)
    min_radius_index = np.argmin(radii)
    min_radius = radii[min_radius_index]
    return data[min_radius_index], min_radius