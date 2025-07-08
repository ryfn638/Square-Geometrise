import CalculateShapes
import random
import math
import pandas as pd
import numpy as np
import time
import CreateShape
import Heatmap

from skimage.color import rgb2lab
from scipy.spatial.distance import cdist


# List of things to do
# Create a generation (First) Done
# Calculating Canvas Accuracy Score
# Choosing Best Canvas Score (object oriented approach maybe?)
# Adjusting Generation based off best score

def randomRange(a,b):
    return random.randrange(a,b)

def constrain(numlist):
    return np.array([max(x, 1) for x in numlist])

def rgb_diff(r1, r2, g1, g2, b1, b2):
    r_diff=r1-r2
    g_diff=g1-g2
    b_diff=b1-b2
    return np.sqrt((r_diff**2)+(g_diff**2)+(b_diff**2))

def ListIndexToCoordinate(width, index_num):
    y = math.floor(index_num/width)
    x = index_num - y*width
    return x,y

def GenerateAccuracyScore(canvas, imagedata):
    canvas_rgb = canvas.flatten().reshape(-1,3)
    imagedata_rgb = imagedata.flatten().reshape(-1,3)
    r_canvas = canvas_rgb[:, 0]
    g_canvas = canvas_rgb[:, 1]
    b_canvas = canvas_rgb[:, 2]

    r_image = imagedata_rgb[:, 0]
    g_image = imagedata_rgb[:, 1]
    b_image = imagedata_rgb[:, 2]

    colour_score = np.sum(rgb_diff(r_canvas, r_image, g_canvas, g_image, b_canvas, b_image))
    return colour_score

def process_shape(x_max, y_max, location, current_canvas, image_data, image, size_scale):
    opacity = []
    angle = []
    x_sizes = []
    y_sizes = []
    colour_list = []
    canvas_list = []
    colour_scores = []


    opacity.append(randomRange(10,255))

    angle.append(random.random()*(math.pi))


    x_sizes.append(1+randomRange(1,1+max(round(max(x_max, y_max)),1))*size_scale)
    y_sizes.append(1+randomRange(1,1+max(round(max(x_max, y_max)),1))*size_scale)

    calc_rectangle_class = CalculateShapes.CalcRectangle(location, x_sizes[-1], y_sizes[-1], angle[-1], opacity[-1], current_canvas, image_data, image)
    calc_rectangle_class.calcCorners()

    canvas_new, avg_rgb = calc_rectangle_class.FilledSpaces()

    colour_list.append(avg_rgb)

    canvas_list.append(canvas_new)
    colour_scores.append(GenerateAccuracyScore(canvas_list[-1], image_data))


class EvolveAlgorithm():
    def __init__(self, canvas, imagedata, image, generations, population, location, size_scale):
        self.canvas = canvas
        self.imagedata = imagedata
        self.generations = generations
        self.population = population
        self.image = image
        self.location = location
        self.size_scale = size_scale

    def CreateGen(self):
        opacity = []
        angle = []
        x_sizes = []
        y_sizes = []
        canvas_list = []
        image=self.image
        image_data = self.imagedata

        colour_list = []
        location = self.location
        canvas_num = 0
        width, height = self.image.size


        location_class = Heatmap.HeatmapLocation(self.canvas, image_data, width, height, self.size_scale)
        location, colour_indices= location_class.calc_area_score()
        location = location[::-1]
        
        opt_x, opt_y, opt_angle = location_class.suggest_sizes()
        
        #print(opt_x, opt_y)
        for currentgen in range(self.generations):
            if currentgen == 0:
                x_max, y_max = image.size
                current_canvas = np.reshape(np.array([x for x in self.canvas]), (y_max,x_max,3))

                # Pre-calculate constants
                max_dim = max(round(max(x_max, y_max)), 1)

                # Vectorized random generation
                opacity = np.random.randint(50, 256, size=self.population)
                #print(angle * 180/np.pi)
                x_sizes = (opt_x * np.array(np.random.uniform(0.1, 1.25, size=self.population))).astype(int)
                x_sizes = constrain(x_sizes)


                y_sizes = (opt_y * np.array(np.random.uniform(0.1, 1.25, size=self.population))).astype(int)
                y_sizes = constrain(y_sizes)

                modded_x_sizes = x_sizes.reshape(-1, 1)
                modded_y_sizes = y_sizes.reshape(-1, 1)
                angle = (np.arctan(modded_y_sizes[:,0]/modded_x_sizes[:,0]) * np.random.uniform(0.8, 1.2, size=self.population))
                

                results = [CreateShape.CreateShape(location, sizeX, sizeY, degree, transparency, current_canvas, image_data, image) for sizeX, sizeY, degree, transparency in zip(x_sizes, y_sizes, angle, opacity)]

                canvas_list, avg_rgb = zip(*results)
                colour_list= avg_rgb
                all_colour_scores = [GenerateAccuracyScore(canvas, image_data) for canvas in canvas_list]
            #print(len(canvas_list))
            best_colour_score = min(all_colour_scores)
            best_shape_index = all_colour_scores.index(best_colour_score)
        size_x = x_sizes[best_shape_index]
        size_y = y_sizes[best_shape_index]
        best_opacity = opacity[best_shape_index]
        best_colour = colour_list[best_shape_index]
        best_angle = angle[best_shape_index]
        best_canvas = canvas_list[best_shape_index]
        return size_x, size_y, best_colour, best_opacity, best_angle, best_canvas, best_colour_score, location



                

                    

