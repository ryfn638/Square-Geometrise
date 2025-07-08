import EvolutionAlgorithm
import CalculateShapes
from multiprocessing import Pool
import time

## Individual Function to define creating a shape. This will be run like 30 times at once lol
def CreateShape(location, x_size, y_size, angle, opacity, current_canvas, image_data, image):
    calc_rectangle_class = CalculateShapes.CalcRectangle(location, x_size, y_size, angle, opacity, current_canvas, image_data, image)
    calc_rectangle_class.calcCorners()

    canvas_list, avg_rgb = calc_rectangle_class.FilledSpaces()

    return canvas_list, avg_rgb

## Generate a generation here. This is placed here to prevent accidentally triggering some other code i dont want
def run_multiprocess(location, x_sizes, y_sizes, angle, opacity, current_canvas, image_data, image):
    params = [(location, sizeX, sizeY, degree, transparency, current_canvas, image_data, image)
              for sizeX, sizeY, degree, transparency in zip(x_sizes, y_sizes, angle, opacity)]

    with Pool(processes=4) as pool:
        results = pool.starmap(CreateShape, params)  # Use starmap here
    canvas_list, avg_rgb = zip(*results)

    return canvas_list, avg_rgb