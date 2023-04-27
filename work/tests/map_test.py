# Not maintained

import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from utils import read_csv

map_id = random.randint(1, 250)

map_png = f"tracks/maps/map{map_id}.png"
map_csv = f"tracks/centerline/map{map_id}.csv"
map_yaml = f"tracks/maps/map{map_id}.yaml"

image = mpimg.imread(map_png)
map_data = np.array(read_csv(map_csv))

with open(map_yaml, "r") as file:
    yaml_data = yaml.safe_load(file)

map_origin = yaml_data["origin"][0:2]
map_resolution = yaml_data["resolution"]

map_x = (map_data[:, 1] - map_origin[0]) / map_resolution
map_y = (map_data[:, 2] - map_origin[1]) / map_resolution

flipped_image = np.flipud(image)
plt.imshow(flipped_image)

plt.plot(map_x, map_y, markersize=1)
plt.show()
