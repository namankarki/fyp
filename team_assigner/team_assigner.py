# from development_and_analysis .k_means_custom import KMeansCustom
# import numpy as np

# class TeamAssigner:
#     def __init__(self):
#         self.team_colors = {}
#         self.player_team_dict = {}

#     def get_clustering_model(self, image):
#         # Reshape the image to a 2D array
#         image_2d = image.reshape(-1, 3)

#         # Perform K-means with 2 clusters using the custom KMeans implementation
#         kmeans = KMeansCustom(n_clusters=2, random_state=0)
#         kmeans.fit(image_2d)

#         return kmeans

#     def get_player_color(self, frame, bbox):
#         # Crop the image to the player's bounding box
#         image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

#         # Use the top half of the bounding box image
#         top_half_image = image[0:int(image.shape[0] / 2), :]

#         # Reshape the image to a 2D array for clustering
#         image_2d = top_half_image.reshape(-1, 3)

#         # Perform clustering with the custom KMeans implementation
#         kmeans = KMeansCustom(n_clusters=2, random_state=0)
#         kmeans.fit(image_2d)

#         # Get the cluster labels for each pixel
#         labels = kmeans.labels

#         # Reshape the labels to the original image shape
#         clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

#         # Determine the player cluster by analyzing the corner clusters
#         corner_clusters = [
#             clustered_image[0, 0],        # Top-left corner
#             clustered_image[0, -1],       # Top-right corner
#             clustered_image[-1, 0],       # Bottom-left corner
#             clustered_image[-1, -1]       # Bottom-right corner
#     ]
#         non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
#         player_cluster = 1 - non_player_cluster

#         # Get the color of the player cluster (centroid of the cluster)
#         player_color = kmeans.centroids[player_cluster]

#         return player_color
    

#     def assign_team_color(self, frame, player_detections):
#         # Initialize an empty list to store player colors
#         player_colors = []

#         # Extract player colors from each player's bounding box
#         for _, player_detection in player_detections.items():
#             bbox = player_detection["bbox"]
#             player_color = self.get_player_color(frame, bbox)  # Use the custom get_player_color method
#             player_colors.append(player_color)

#         # Perform clustering on player colors using the custom KMeans implementation
#         kmeans = KMeansCustom(n_clusters=2, random_state=0)
#         kmeans.fit(np.array(player_colors))  # Convert to a NumPy array if not already

#         # Store the kmeans instance for later use
#         self.kmeans = kmeans

#         # Assign team colors based on cluster centers
#         self.team_colors[1] = kmeans.centroids[0]
#         self.team_colors[2] = kmeans.centroids[1]
        
#     def get_player_team(self, frame, player_bbox, player_id):
#         # Check if the player's team is already assigned
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]

#         # Get the player's color from their bounding box
#         player_color = self.get_player_color(frame, player_bbox)

#         # Predict the player's team using the custom KMeans
#         player_color = np.array(player_color).reshape(1, -1)  # Reshape to 2D array for prediction
#         team_id = self.kmeans.predict(player_color)[0]  # Use the custom predict method

#         # Adjust team ID to be 1-based
#         team_id += 1

#         # Handle special case for player ID 91
#         if player_id == 91:
#             team_id = 1

#         # Store the player's team assignment
#         self.player_team_dict[player_id] = team_id

#         return team_id

from utils import get_top_half, reshape_to_2d, perform_kmeans, analyze_corners
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Maps team IDs to colors
        self.player_team_dict = {}  # Maps player IDs to team IDs

    def get_clustering_model(self, image):
        """Perform K-Means clustering on the entire image."""
        # Reshape the image to a 2D array
        image_2d = reshape_to_2d(image)

        # Perform K-means clustering
        kmeans = perform_kmeans(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        """Extract the dominant color of a player's uniform."""
        # Crop the image to the player's bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Use the top half of the bounding box image
        top_half_image = get_top_half(image)

        # Reshape the image to a 2D array for clustering
        image_2d = reshape_to_2d(top_half_image)

        # Perform clustering
        kmeans = perform_kmeans(image_2d)

        # Get the cluster labels and reshape them
        clustered_image = kmeans.labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Determine the player cluster by analyzing the corners
        player_cluster, _ = analyze_corners(clustered_image)

        # Get the color of the player cluster (centroid of the cluster)
        player_color = kmeans.centroids[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        """Assign colors to teams based on detected player colors."""
        # Initialize an empty list to store player colors
        player_colors = []

        # Extract player colors from each player's bounding box
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Perform clustering on player colors
        kmeans = perform_kmeans(np.array(player_colors))

        # Store the kmeans instance for later use
        self.kmeans = kmeans

        # Assign team colors based on cluster centers
        self.team_colors[1] = kmeans.centroids[0]
        self.team_colors[2] = kmeans.centroids[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """Determine the team of a player based on their uniform color."""
        # Check if the player's team is already assigned
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the player's color from their bounding box
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the player's team using the custom KMeans
        player_color = np.array(player_color).reshape(1, -1)  # Reshape to 2D array
        team_id = self.kmeans.predict(player_color)[0]  # Predict the cluster

        # Adjust team ID to be 1-based
        team_id += 1

        # Handle special case for player ID 91
        # if player_id == 91:
        #     team_id = 1

        # Store the player's team assignment
        self.player_team_dict[player_id] = team_id

        return team_id

