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
from tqdm import tqdm

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
# target_latitudes = [target['latitude'] for target in data['targets']]
# target_longitudes = [target['longitude'] for target in data['targets']]
# ax.scatter(target_longitudes, target_latitudes, transform=ccrs.Geodetic(), color='black', s=10, zorder=3, label='Targets')

# Plot the targets
target_latitudes = [target['latitude'] for target in data['targets']]
target_longitudes = [target['longitude'] for target in data['targets']]
targets_scatter = ax.scatter(target_longitudes, target_latitudes, transform=ccrs.Geodetic(), color='black', s=10, zorder=3, label='Targets')

# Get satellite names from the data
satellite_names = list(data['steps'][0].keys())
# satellite_colors = {sat: c for sat, c in zip(satellite_names, ['blue', 'green', 'purple'])}
# We now have 100 + satellites so we need to assign colors
num_satellites = len(satellite_names)
colors = plt.cm.viridis(np.linspace(0, 1, num_satellites))
satellite_colors = {sat: colors[i] for i, sat in enumerate(satellite_names)}




# Prepare satellite data
satellites = {sat: {'times': [], 'lats': [], 'lons': [], 'actions': [], 'task_being_collected': [], 'rewards': []} for sat in satellite_colors}

# Add a new dictionary for rewards
satellite_rewards = {sat: 0 for sat in satellite_colors}
collected_targets = set()

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
    scatter = ax.scatter([], [], transform=ccrs.Geodetic(), color=color, s=80, zorder=4)  # Removed label
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

# Add this after initializing collected_targets
targets_being_collected = set()
red_targets = set()


def update(frame, pbar):
    imaging_boxes.set_paths([])
    reward_string = ""
    current_targets_being_collected = set()

    for sat, scatter in scatters.items():
        if sat in interpolated_satellites:
            sat_data = interpolated_satellites[sat]
            scatter.set_offsets([[sat_data['lons'][frame], sat_data['lats'][frame]]])
            
            step_index = int(frame * (len(sat_data['actions']) - 1) / (num_interpolation_points - 1))
            action = sat_data['actions'][step_index]
            
            if step_index < len(sat_data['rewards']):
                satellite_rewards[sat] += sat_data['rewards'][step_index]
            reward_string = f"Score {sum(satellite_rewards.values()):.2f}"
            
            if 0 <= action <= 9:
                task = sat_data['task_being_collected'][step_index]

                if task is not None:
                    target_id = (task['longitude'], task['latitude'])
                    current_targets_being_collected.add(target_id)

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

    # Update target colors and remove collected targets
    current_targets = targets_scatter.get_offsets()
    updated_targets = []
    updated_colors = []
    for target in current_targets:
        target_tuple = tuple(target)
        if target_tuple in current_targets_being_collected:
            updated_targets.append(target)
            updated_colors.append('red')
        elif target_tuple not in collected_targets:
            updated_targets.append(target)
            updated_colors.append('black')
        elif target_tuple in collected_targets and target_tuple not in current_targets_being_collected:
            # Target has been collected and is no longer being collected, so we remove it
            continue
        
    targets_scatter.set_offsets(updated_targets)
    targets_scatter.set_facecolors(updated_colors)

    # Update collected_targets
    collected_targets.update(current_targets_being_collected)

    reward_text.set_text(reward_string)

    # Update the progress bar
    pbar.update(1)

    return list(scatters.values()) + [imaging_boxes] + [targets_scatter] + [reward_text]


# # Create the animation
# num_frames = num_interpolation_points
# anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
# # Show the animation
# plt.show()

# # Create the animation
num_frames = num_interpolation_points

num_frames = 100
# anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# # Save the animation
# anim.save('satellite_animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

# plt.close(fig)  # Close the figure to free up memory

# print("Animation saved as 'satellite_animation.mp4'")

# Create a progress bar
pbar = tqdm(total=num_frames, desc="Generating animation")

# Create the animation
anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False, fargs=(pbar,))

# Save the animation
anim.save('satellite_animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

# Close the progress bar
pbar.close()

plt.close(fig)  # Close the figure to free up memory

print("Animation saved as 'satellite_animation.mp4'")