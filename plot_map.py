import json
# import matplotlib
# matplotlib.use('Qt5Agg')  # or 'Qt5Agg' if you prefer Qt
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

# Create a new figure and axis with a map projection
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Plot the targets
target_latitudes = [target['latitude'] for target in data['targets']]
target_longitudes = [target['longitude'] for target in data['targets']]
ax.scatter(target_longitudes, target_latitudes, transform=ccrs.Geodetic(), color='black', s=10, zorder=3, label='Targets')

# Get satellite names from the data
satellite_names = list(data['steps'][0].keys())
satellite_colors = {sat: c for sat, c in zip(satellite_names, ['blue', 'green', 'purple'])}

# Prepare satellite data
satellites = {sat: {'times': [], 'lats': [], 'lons': [], 'actions': [], 'task_being_collected': [], 'rewards': []} for sat in satellite_colors}

# Add a new dictionary for rewards
satellite_rewards = {sat: 0 for sat in satellite_colors}

# Sort steps by time and extract satellite data
sorted_steps = sorted(data['steps'], key=lambda x: list(x.values())[0]['time'])
for step in sorted_steps:
    for sat in satellites:
        if sat in step:
            satellites[sat]['times'].append(step[sat]['time'])
            satellites[sat]['lats'].append(step[sat]['latitude'])
            satellites[sat]['lons'].append(step[sat]['longitude'])
            satellites[sat]['actions'].append(step[sat]['actions'])
            satellites[sat]['task_being_collected'].append(step[sat]['task_being_collected'])
            satellites[sat]['rewards'].append(step[sat]['reward'])

# Interpolate satellite positions
interpolated_satellites = {}
num_interpolation_points = len(sorted_steps) * 10

for sat, data in satellites.items():
    if len(data['times']) > 1:
        times = np.array(data['times'])
        lats = np.array(data['lats'])
        lons = np.array(data['lons'])
        
        # Remove duplicates
        unique_times, unique_indices = np.unique(times, return_index=True)
        unique_lats = lats[unique_indices]
        unique_lons = lons[unique_indices]
        
        if len(unique_times) > 1:
            # Create interpolation functions
            lat_interp = interpolate.interp1d(unique_times, unique_lats, kind='linear')
            lon_interp = interpolate.interp1d(unique_times, unique_lons, kind='linear')
            
            # Generate interpolated points
            interp_times = np.linspace(unique_times.min(), unique_times.max(), num_interpolation_points)
            interp_lats = lat_interp(interp_times)
            interp_lons = lon_interp(interp_times)
            
            interpolated_satellites[sat] = {
                'times': interp_times,
                'lats': interp_lats,
                'lons': interp_lons,
                'actions': data['actions'],
                'task_being_collected': data['task_being_collected'],
                'rewards': data['rewards']
            }

# Create scatter plots for each satellite
scatters = {}
for sat, color in satellite_colors.items():
    scatter = ax.scatter([], [], transform=ccrs.Geodetic(), color=color, s=80, zorder=4, label=f'{sat.split("_")[0]} position')
    scatters[sat] = scatter

# Set the extent of the map
ax.set_global()

# Add a title and legend
plt.title('Target Locations and Satellite Positions')
plt.legend()

# Create a collection for imaging boxes
imaging_boxes = PathCollection([], facecolors='yellow', edgecolors='red', alpha=0.3, linewidths=2, transform=ccrs.PlateCarree())
ax.add_collection(imaging_boxes)

# Create a dictionary to store active imaging lines for each satellite
active_imaging_lines = {sat: None for sat in satellite_colors}

# Add a text object to display rewards
reward_text = ax.text(0.02, 0.98, '', 
                      transform=ax.transAxes, 
                      verticalalignment='top', 
                      fontweight='bold', 
                      fontsize=14)
def update(frame):
    imaging_boxes.set_paths([])
    reward_string = ""
    for sat, scatter in scatters.items():
        if sat in interpolated_satellites:
            sat_data = interpolated_satellites[sat]
            scatter.set_offsets([[sat_data['lons'][frame], sat_data['lats'][frame]]])
            
            # Check if we're at a step with action data
            step_index = int(frame * (len(sat_data['actions']) - 1) / (num_interpolation_points - 1))
            action = sat_data['actions'][step_index]
            
            # Update reward
            if step_index < len(sat_data['rewards']):
                satellite_rewards[sat] += sat_data['rewards'][step_index]
            reward_string = f"Score {satellite_rewards[sat]:.2f}"
            
            if 0 <= action <= 9:
                # target = sat_data['targets'][step_index][action]
                task = sat_data['task_being_collected'][step_index]

                if task is None:
                    continue
                
                # Create imaging box
                box_size = 10  # degrees
                verts = [
                    (task['longitude'] - box_size/2, task['latitude'] - box_size/2),
                    (task['longitude'] + box_size/2, task['latitude'] - box_size/2),
                    (task['longitude'] + box_size/2, task['latitude'] + box_size/2),
                    (task['longitude'] - box_size/2, task['latitude'] + box_size/2),
                    (task['longitude'] - box_size/2, task['latitude'] - box_size/2),
                ]
                codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
                path = Path(verts, codes)
                imaging_boxes.set_paths(imaging_boxes.get_paths() + [path])
                
                # Draw or update line from satellite to target
                if active_imaging_lines[sat] is None:
                    line = ax.plot([sat_data['lons'][frame], task['longitude']], 
                                   [sat_data['lats'][frame], task['latitude']], 
                                   color='black', alpha=0.7, linewidth=4, linestyle=':', transform=ccrs.Geodetic())[0]
                    active_imaging_lines[sat] = line
                else:
                    active_imaging_lines[sat].set_data([sat_data['lons'][frame], task['longitude']], 
                                                       [sat_data['lats'][frame], task['latitude']])
            else:
                # Remove the line if the satellite is not imaging
                if active_imaging_lines[sat] is not None:
                    active_imaging_lines[sat].remove()
                    active_imaging_lines[sat] = None

    # Update reward text
    reward_text.set_text(reward_string)

    return list(scatters.values()) + [imaging_boxes] + [line for line in active_imaging_lines.values() if line is not None] + [reward_text]

# Create the animation
num_frames = num_interpolation_points
anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# Show the animation
plt.show()