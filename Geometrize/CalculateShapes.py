import numpy as np
from PIL import Image
import math
import random
import time

def randomRange(a,b):
    return random.randrange(a,b)

def power(num, power):
    return num ** power

def DistancebetweenPoints(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.sqrt(power(x2-x1, 2)+power(y2-y1,2))

def ListIndexToCoordinate(width, index_num):
    y = math.floor(index_num/width)
    x = index_num - y*width
    return x,y

def CoordinatetoListIndex(x,y,y_min,width):
    return ((y-y_min-1)*width + x)

def blendRGB(new,old,opacity):
    return ((opacity/255) * new + (1-(opacity/255))*old)


def AddColour(canvas, colour, opacity, indices):
    canvas_copy = np.array(canvas)

    y_indices = indices[:, 0]
    x_indices = indices[:, 1]

    # Retrieve the colors at the specified indices
    selected_pixels = canvas[y_indices, x_indices]
    r = blendRGB(colour[0], selected_pixels[:, 0], opacity)
    g = blendRGB(colour[1], selected_pixels[:, 1], opacity)
    b = blendRGB(colour[2], selected_pixels[:, 2], opacity)

    canvas_copy[y_indices, x_indices] = np.stack([r, g, b], axis=1)

    return canvas_copy
    

def findHighestNum(array):
    maxim = max(array)
    return array.index(maxim)

def findSmallestNum(array):
    maxim = min(array)
    return array.index(maxim)

def area(x1, y1, x2, y2, x3, y3):
    # Use vectorized operations for area computation
    return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

def isInside(x1, y1, x2, y2, x3, y3, x, y):
    # Calculate the area of the triangle formed by the vertices (x1, y1), (x2, y2), and (x3, y3)
    A = area(x1, y1, x2, y2, x3, y3)
    
    # Calculate the area of the triangle formed by (x, y), (x2, y2), and (x3, y3)
    A1 = area(x, y, x2, y2, x3, y3)
    
    # Calculate the area of the triangle formed by (x1, y1), (x, y), and (x3, y3)
    A2 = area(x1, y1, x, y, x3, y3)
    
    # Calculate the area of the triangle formed by (x1, y1), (x2, y2), and (x, y)
    A3 = area(x1, y1, x2, y2, x, y)
    
    # Compare the total area with the sum of the sub-triangle areas
    return A == A1 + A2 + A3
# Default is Rectangle


class CalcRectangle():
    def __init__(self,center_point, sizex, sizey, angle, opacity, canvas, image_data, image):
        self.center_point = center_point
        self.image_data = image_data
        self.sizeX = sizex
        self.sizeY = sizey
        self.rectangle_angle = angle
        self.opacity = opacity
        self.canvas = canvas
        self.img = image



    
    def calcCorners(self):
        img_x, img_y = self.img.size

        # no optimisations needed
        size_x = self.sizeX
        size_y = self.sizeY
        top_left_corner = self.center_point
        img_data = self.image_data

        center_to_side = size_x/2
        center_to_top = size_y/2

        center_to_corner = np.sqrt(power(center_to_side, 2)+power(center_to_top, 2))

        default_angle = np.arctan(center_to_top/center_to_side)
        angle_offset = self.rectangle_angle

        center_point = top_left_corner

        horizontal_diff_1 = center_to_corner * np.cos(angle_offset-default_angle)
        horizontal_diff_2 = center_to_corner * np.cos(angle_offset+default_angle)

        vertical_diff_1 = center_to_corner * np.sin(angle_offset-default_angle)
        vertical_diff_2 = center_to_corner * np.sin(angle_offset+default_angle)

        top_right_corner = np.array([
            np.clip(np.round(center_point[0]+horizontal_diff_1), 0, img_x),
            np.clip(np.round(center_point[1]-vertical_diff_1), 0, img_y)
            ])
        bottom_right_corner = np.array([
            np.clip(np.round(center_point[0]+horizontal_diff_2), 0, img_x),
            np.clip(np.round(center_point[1]-vertical_diff_2), 0, img_y)
            ])
        
        bottom_left_corner = np.array([
            np.clip(np.round(center_point[0]-horizontal_diff_1), 0, img_x),
            np.clip(np.round(center_point[1]+vertical_diff_1), 0, img_y)
            ])
        
        top_left_corner = np.array([
            np.clip(np.round(center_point[0]-horizontal_diff_2), 0, img_x), 
           np.clip(np.round(center_point[1]+vertical_diff_2), 0, img_y)
            ])
        
        
        rotated_corners = [
            top_right_corner,
            bottom_right_corner,
            bottom_left_corner,
            top_left_corner
        ]
        #print(rotated_corners)
        self.corners = rotated_corners


    # Get the filled spaces in the shape chosen, and fill the canvas colour with said shapes
    def FilledSpaces(self):
        # Get image dimensions
        img_x, img_y = self.img.size
    
        # Extract corner coordinates
        [top_right_corner, bottom_right_corner, bottom_left_corner, top_left_corner] = self.corners
        corner_x = np.array([top_right_corner[0], bottom_right_corner[0], bottom_left_corner[0], top_left_corner[0]])
        corner_y = np.array([top_right_corner[1], bottom_right_corner[1], bottom_left_corner[1], top_left_corner[1]])

        # Minimize grid size for efficiency

        x_min = int(np.floor(np.min(corner_x)))
        x_max = int(np.ceil(np.max(corner_x)))

        y_min = int(np.floor(np.min(corner_y)))
        y_max = int(np.ceil(np.max(corner_y)))

        # Create canvas grid
        canva = self.canvas.reshape(img_y, img_x, 3)
    
        # Generate grid
        y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]

        # Generate masks
        mask1 = isInside(corner_x[0], corner_y[0], 
                     corner_x[1], corner_y[1], 
                     corner_x[3], corner_y[3], 
                     x_grid, y_grid)
        
        mask2 = isInside(corner_x[2], corner_y[2], 
                        corner_x[1], corner_y[1], 
                        corner_x[3], corner_y[3], 
                        x_grid, y_grid)

        masks = mask1 | mask2  # Combine masks
        
        # Get important indices
        masked_points = np.where(masks)
        important_indices = np.column_stack((masked_points[0] + y_min, masked_points[1] + x_min))
        # Extract image data values at important indices
        image_data_values = self.image_data[important_indices[:, 0], important_indices[:, 1]]
        
        avg_rgb = image_data_values.mean(axis=0).astype(int)
        #avg_rgb = self.image_data[self.center_point[1], self.center_point[0]]
        
        # Apply color changes
        complete_canvas = AddColour(canva, avg_rgb, self.opacity, important_indices)

        return complete_canvas, avg_rgb
    