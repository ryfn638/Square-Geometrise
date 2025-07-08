import pyglet
from pyglet import shapes
import EvolutionAlgorithm as Evolution
import Heatmap
import math
from PIL import Image
import numpy as np
from pyglet.window import Window
from pyglet.app import run
from pyglet.window import FPSDisplay
from pyglet.graphics import Batch
from pyglet import clock


# AI Components
import numpy as np

from skimage.color import rgb2lab

if __name__ == "__main__":
    current_shape_index = 0
    img = Image.open(r"C:\Users\ryanf\Desktop\beverage\Geometrize\Images\birdsmal.png")
    size_multiplier = 1
    width, height = img.size
    window = pyglet.window.Window(width*size_multiplier, height*size_multiplier)
    batch = pyglet.graphics.Batch()

    img_data = img.getdata()

def canvas_avg_colour(pixel_data):
    r,g,b = 0,0,0
    for num in pixel_data:
        r += num[0]
        g += num[1]
        b += num[2]
    r = round(r/len(pixel_data))
    g = round(g/len(pixel_data))
    b = round(b/len(pixel_data))
    return (r,g,b)

def ListIndexToCoordinate(width, index_num):
    y = math.floor(index_num/width)
    x = index_num - y*width
    return x,y


def rgb_diff(rgb1, rgb2):
    r_diff=rgb1[0]-rgb2[0]
    g_diff=rgb1[1]-rgb2[1]
    b_diff=rgb1[2]-rgb2[2]
    return math.sqrt((r_diff**2)+(g_diff**2)+(b_diff**2))

def rgb_diff2(r1, r2, g1, g2, b1, b2):
    r_diff=r1-r2
    g_diff=g1-g2
    b_diff=b1-b2
    return np.sqrt((r_diff**2)+(g_diff**2)+(b_diff**2))

def DifferenceHeatmap(canvas, imagedata, height, width):
    # Ensure the input dimensions match
    image_pixel_copy = imagedata
    lab_colors = rgb2lab(image_pixel_copy / 255)
    lab_colors = lab_colors.flatten().reshape(-1, 3)

    canvas = canvas.flatten().reshape(-1, 3)


    lab_canvas = rgb2lab(canvas / 255)
        
    # Calculate the color difference between the canvas and image
    canvas_error = np.sqrt(np.linalg.norm((lab_colors - lab_canvas) ** 2, axis=1)).reshape(height, width)
    max_location = np.where(canvas_error == np.max(canvas_error))
    return (max_location[0][0], max_location[1][0])


def GenerateAccuracyScore(canvas, imagedata, width, height):
    colour_score = 0
    for col in range(width*height):
        x, y = ListIndexToCoordinate(width, col)
        colour_score += rgb_diff(canvas[y,x], imagedata[y,x])
    return colour_score


# The idea is that we take a larger image, shrink it down for general detail, and as the image gradually requires more specific shapes we slowly upscale it so the features arent reduced.


class MainRun():
    def __init__(self):
        self.default_col_score = 0
    def main_run(self):
        if __name__ == "__main__":
            pixels_img = [x for x in img_data]
            #avg_rgb = canvas_avg_colour(pixels_img)
            pixels_img = np.reshape(np.array(pixels_img, dtype=int), (height,width,3))
            pixels_img = np.flip(pixels_img, 0)
            canvas = np.reshape(np.array([[0,0,0] for x in img_data], dtype=int), (height,width,3))

            self.height = height
            self.width = width
            self.image = pixels_img

            global rectangles
            rectangles = []
            #rectangles.append(shapes.Rectangle(0,0,width*size_multiplier, height*size_multiplier, color=avg_rgb, batch=batch))

            generations = 1
            population = 100

            num_of_shapes = 300
            x_max = width
            y_max = height

            # Next Idea: Create a list of locations with their colour score, and the locations with the highest colour score will have the next shape placed there
            #CustomEnv(target_image=pixels_img, default_score=default_colour_score, canvas, width, height, image)

            self.default_col_score = GenerateAccuracyScore(canvas, pixels_img, width, height)

            for list in range(num_of_shapes):

                most_critical_spot = DifferenceHeatmap(canvas, pixels_img, height, width)
                ### Identify any black spots on the canvas, and consequently allocate those as a priority.
                #x_black, y_black = np.where(np.all(canvas == [0,0,0], axis=1))
                # ideally as time goes on smaller shapes naturally get chosen, this is just to thin out unnecessary large shapes on shape 500 or so
                size_scale = np.min((30*round(np.sqrt(list/10))+5, 150))
                #size_scale = 30
                self.canvas = canvas
                evolve = Evolution.EvolveAlgorithm(canvas, pixels_img, img, generations, population, most_critical_spot, size_scale)
                shape = evolve.CreateGen()
                center_to_corner = math.sqrt(((shape[0]/2)**2)+((shape[1]/2)**2))
                default_angle = math.asin((shape[1]/2)/center_to_corner)
                
                new_angle = default_angle - shape[4]
                location = shape[7]

                #print(location, most_critical_spot[::-1])
                #x_diff = center_to_corner * math.cos(new_angle)
                x_diff = 0
                #y_diff = center_to_corner * math.sin(new_angle)
                y_diff = 0
                rectangles.append(shapes.Rectangle((location[0]-x_diff)*size_multiplier,(location[1]-y_diff)*size_multiplier,shape[0]*size_multiplier, shape[1]*size_multiplier, color=(shape[2]), batch=batch))
                rectangles[-1].opacity = shape[3]
                rectangles[-1].anchor_x = rectangles[-1].width//2
                rectangles[-1].anchor_y = rectangles[-1].height//2
                rectangles[-1].rotation = shape[4]*180/math.pi

                canvas = shape[5]
                default_colour_score = shape[6]
                print(str(round((list+1)/num_of_shapes*100,2)) + "% ", end="\r")
            return rectangles


def draw_shape(shape):
    shape.draw()

if __name__ == "__main__":
    @window.event
    def on_draw():
        #window.clear()
        if current_shape_index > 0:
            draw_shape(all_shapes[current_shape_index-1])


# Animation of drawing all shapes one by one
def update(dt):
    global current_shape_index
    if current_shape_index < len(all_shapes):
        # Clear the window before drawing
        #window.clear()
        # Draw the current shape
        draw_shape(all_shapes[current_shape_index])
        # Move to the next shape
        current_shape_index += 1


if __name__ == "__main__":
    global all_shapes
    main_run_class = MainRun()
    all_shapes = main_run_class.main_run()
    pyglet.clock.schedule_interval(update, 0.086633)

    pyglet.app.run()