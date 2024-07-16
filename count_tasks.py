import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy import interpolate
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection
from matplotlib.path import Path
import matplotlib.transforms as mtransforms

# Load the JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

targets = set()
all_targets = list()

for step in data['steps']:

    for sat_id, sat in step.items():
        action = sat['actions']
        if action < 9:
            target = sat['targets'][action]
            targets.add(target['id'])
            all_targets.append(target)

print(f"Number of targets: {len(targets)}")
print(f"Number of total targets collected: {len(all_targets)}")
