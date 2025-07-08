import numpy as np
import math
from skimage.color import rgb2lab
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from scipy import stats as st
import time
##
##
## This script is for calcualting the most optimal location for a shape to go
## Checklist for this code
## Find the median area of the colour score region, prioritise area and not singular colour score values
##
def sigmoid(x):
    return 1/(1+np.e**(-(x-8)))
    #return x

def rgb_diff(r1, r2, g1, g2, b1, b2):
    r_diff=r1-r2
    g_diff=g1-g2
    b_diff=b1-b2
    return np.sqrt((r_diff**2)+(g_diff**2)+(b_diff**2))

def coordinate_to_index(x, y, width):
    return y * width + x


def calculateMiddlePoint(points, highest_point):
    # Step 1: Apply DBSCAN clustering
    eps = 1  # Maximum distance between points to be considered a cluster
    min_samples = 3  # Minimum number of points to form a cluster
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # Extract labels
    labels = db.labels_
    highest_index = np.where((points == highest_point).all(axis=1))[0]
    highest_label = (labels[highest_index])
    
    connected_points = points[np.where(labels == highest_label)]
    return connected_points

class HeatmapLocation():
    def __init__(self, canvas, image_data, width, height, threshold_scale):
        self.image_data = image_data
        self.canvas = canvas
        self.width = width
        self.height = height
        self.scale = threshold_scale

    def GenerateAccuracyScore(self, canvas, imagedata):
        canvas_rgb = canvas.flatten().reshape(-1,3)
        imagedata_rgb = imagedata.flatten().reshape(-1,3)
        r_canvas = canvas_rgb[:, 0]
        g_canvas = canvas_rgb[:, 1]
        b_canvas = canvas_rgb[:, 2]

        r_image = imagedata_rgb[:, 0]
        g_image = imagedata_rgb[:, 1]
        b_image = imagedata_rgb[:, 2]

        colour_score = rgb_diff(r_canvas, r_image, g_canvas, g_image, b_canvas, b_image)
        return colour_score


    ## Calculate all individual colours
    def get_unique_scores(self):
        time1 = time.time()
        image_pixel_copy = self.image_data
        self.lab_colors = rgb2lab(image_pixel_copy / 255)
        lab_colors = self.lab_colors.flatten().reshape(-1, 3)

        canvas = self.canvas
        canvas = canvas.flatten().reshape(-1, 3)


        lab_canvas = rgb2lab(canvas / 255)
        
        y, x = np.meshgrid(range(self.height), range(self.width), indexing='ij')
        spatial_data = np.stack((x.ravel(), y.ravel()), axis=-1)
        #spatial_data = spatial_data / max(self.height, self.width)  # Normalize spatial data to [0, 1]
        features = np.concatenate((spatial_data, lab_colors), axis=1)

        # Calculate the color difference between the canvas and image
        self.canvas_error = np.sqrt(np.linalg.norm((lab_colors - lab_canvas) ** 2, axis=1)).reshape(self.height, self.width)
        canvas_error = self.canvas_error.reshape(-1, 1)
        self.img_converted_lab = lab_colors

        feature = np.concatenate((features, canvas_error), axis=1)
        #Normalise the canvas area
        # Apply DBSCAN clustering
        clustering = MiniBatchKMeans(n_clusters=self.scale, batch_size=4096, max_iter=100, random_state=42).fit(feature)
        labels = clustering.labels_
        unique_labels = np.unique(labels)

        label_scores = [np.sum(self.canvas_error[(feature[np.where(label == labels)][:,1]).astype(int), (feature[np.where(label == labels)][:,0]).astype(int)]) for label in range(unique_labels.size)]
        most_common_label = np.where(label_scores == np.max(label_scores))
        x_coords = feature[np.where(most_common_label[0] == labels)][:,0]
        y_coords = feature[np.where(most_common_label[0] == labels)][:,1]
        indices = np.column_stack((y_coords, x_coords)).astype(int)
        self.highest_point = indices[int(len(indices)/2)]
        #print(time.time() - time1)

        return indices

    def ListIndexToCoordinate(self, width, index_num):
        y = math.floor(index_num/width)
        x = int(index_num - y*width)
        return x,y

    ## Get The Zone Score. Place the shape based on point density and not the median area. although theoretically the highest density should have the median, this doesnt always work and one shit shape snowballs everything
    def calc_area_score(self):
        most_important_indexes = self.get_unique_scores()
        self.significant_indices = most_important_indexes
        #mid_point, self.size= calculateMiddlePoint(most_important_indexes)
        #self.significant_indices = calculateMiddlePoint(most_important_indexes, self.highest_point)

        x_coords = self.significant_indices[:, 1]
        y_coords = self.significant_indices[:, 0]

        canv_error = self.canvas_error[y_coords, x_coords].reshape(-1, 1)
        
        #mean = np.mean(canv_error)
        #std_dev = np.std(canv_error)
        #feature = np.concatenate((self.significant_indices, self.lab_colors[self.significant_indices[:, 0], self.significant_indices[:, 1]].reshape(-1, 3)), axis=1)
        #final_scan = DBSCAN(eps=4, min_samples=2).fit(feature)
        #labels = final_scan.labels_
        #highest_index = np.where((self.significant_indices == self.highest_point).all(axis=1))[0]
        #highest_label = (labels[highest_index])
        #feature_x_coords = feature[np.where(labels == highest_label)][:, 1]
        #feature_y_coords = feature[np.where(labels == highest_label)][:, 0]

        #self.significant_indices = np.array([[y, x] for x, y in zip(feature_x_coords, feature_y_coords)]).astype(int)
        median_x = np.median(x_coords).astype(int)
        median_y = np.median(y_coords).astype(int)
        # Combine into a single point
        mid_point = (median_y, median_x)
        #mid_point = self.highest_point
        #mid_point = np.median(most_important_indexes)
        #self.loc = mid_point
        return mid_point, self.significant_indices
    

    ## Extension function for convenience
    def suggest_sizes(self):
        optimal_size = CalcSizes(self.significant_indices)

        return optimal_size.suggest_size()


class CalcSizes():
    def __init__(self, indices):
        self.indices = indices
    
    def suggest_size(self):

        x_coords = self.indices[:, 1]
        y_coords = self.indices[:, 0]
        max_x = np.max(x_coords)
        min_x = np.min(x_coords)

        max_y = np.max(y_coords)
        min_y = np.min(y_coords)

        opt_x = (max_x - min_x+1)
        opt_y = (max_y - min_y+1)

        asp_ratio_x = opt_x/opt_y
        asp_ratio_y = opt_y/opt_x

        opt_x = (np.sqrt(self.indices.size) * asp_ratio_x) + 1
        opt_y = (np.sqrt(self.indices.size) * asp_ratio_y) + 1
        opt_angle = np.arctan(opt_x/opt_y) - np.arctan(1/1)
        return opt_x, opt_y, opt_angle


