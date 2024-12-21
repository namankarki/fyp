import numpy as np
from development_and_analysis.k_means_custom import KMeansCustom

def get_top_half(image):
    """Extract the top half of the given image."""
    return image[0:int(image.shape[0] / 2), :]

def reshape_to_2d(image):
    """Reshape the image to a 2D array of pixels."""
    return image.reshape(-1, 3)

def perform_kmeans(image_2d, n_clusters=2, random_state=0):
    """Perform K-Means clustering on the given 2D image data."""
    kmeans = KMeansCustom(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(image_2d)
    return kmeans

def analyze_corners(clustered_image):
    """Analyze the corner clusters to identify the player cluster."""
    corner_clusters = [
        clustered_image[0, 0],
        clustered_image[0, -1],
        clustered_image[-1, 0],
        clustered_image[-1, -1]
    ]
    non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
    player_cluster = 1 - non_player_cluster
    return player_cluster, non_player_cluster
