import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy import interpolate
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from tqdm import tqdm
import argparse
import time


class Target:

    def __init__(self, idx, id, lat, lon, required_collects, markers=['o', 's', '^'], colors=['black', 'black', 'black']):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.required_collects = required_collects
        self.idx = idx
        self.markers = markers
        self.colors = colors
        self.reset()

        
    def set_collecting(self, task_reward):
        self.color = 'red'
        self.collecting_count += 1
        if task_reward> 0:
            self.color = 'green'
        
    def reset(self):
        self.color = self.colors[self.required_collects - 1]
        self.marker = self.markers[self.required_collects - 1]
        self.collecting_count = 0


class Targets:
    def __init__(self, ax):
        self.targets = {}
        self.ax = ax
        self.scatters = {}

        self.markers = ['o', 's', '^']  # circle for 1, square for 2, triangle for 3+
        self.colors = ['black', 'black', 'black']

    def reset_scatters(self):
        scatter_targets = {}
        self.scatters = {m: self.ax.scatter([], [], transform=ccrs.Geodetic(), color=c, s=10, zorder=4, marker=m, label=f'{i + 1}  Collects') for i, (m, c) in  enumerate(zip(self.markers, self.colors))}
        scatter_targets = {m: {'offsets': [], 'colors': []} for m in self.markers}

        for target in self.targets.values():
            scatter_targets[target.marker]['offsets'].append((target.lon, target.lat))
            scatter_targets[target.marker]['colors'].append(target.color)

        for marker, data in scatter_targets.items():
            if len(data['offsets']) == 0:
                continue
            self.scatters[marker].set_offsets(data['offsets'])
            self.scatters[marker].set_facecolors(data['colors'])

            
    def render(self):
        self.reset_scatters()

    def add_target(self, id, lat, lon, required_collects):
        print("Adding target required_collects: ", required_collects)
        self.targets[id] = Target(len(self.targets), id, lat, lon, required_collects)
        
    def get_target(self, target_id):
        return self.targets[target_id]
    
    def get_targets(self):
        return self.targets.values()

    def reset(self):
        for target in self.targets.values():
            target.reset()

    def get_artists(self):
        return list(self.scatters.values())

class SatelliteVisualization:
    def __init__(self, data_file, draw_lines=False, interpolation_points=50):
        self.data = self.load_data(data_file)
        self.fig, self.ax = self.setup_map()
        self.satellites = self.prepare_satellite_data()
        self.targets = self.prepare_target_data()
        self.interpolated_satellites = self.interpolate_satellite_positions(interpolation_points)
        self.draw_lines = draw_lines
        self.collection_lines = []
        self.setup_visualization()

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def setup_map(self):
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()
        return fig, ax

    def prepare_satellite_data(self):
        satellite_names = list(self.data['steps'][0].keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(satellite_names)))
        return {sat: {'color': colors[i], 'data': self.extract_satellite_data(sat)} 
                for i, sat in enumerate(satellite_names)}

    def extract_satellite_data(self, satellite):
        return {
            'times': [], 'lats': [], 'lons': [], 
            'actions': [], 'task_being_collected': [], 'rewards': [], 'task_reward': []
        }

    def prepare_target_data(self):
        targets = Targets(self.ax)
        for target in self.data['targets']:
            targets.add_target(target['id'], target['latitude'], target['longitude'], target['simultaneous_collects_required'])
        return targets

    def interpolate_satellite_positions(self, num_interpolation_points):
        interpolated = {}
        sorted_steps = sorted(self.data['steps'], key=lambda x: list(x.values())[0]['time'])
        num_interpolation_points = len(sorted_steps) * num_interpolation_points

        for sat, sat_data in self.satellites.items():
            self.populate_satellite_data(sat, sorted_steps)
            times = np.array(sat_data['data']['times'])
            lats = np.array(sat_data['data']['lats'])
            lons = np.array(sat_data['data']['lons'])

            if len(times) > 1:
                # Handle longitude wrap-around
                lon_diff = np.diff(lons)
                lon_diff[lon_diff > 180] -= 360
                lon_diff[lon_diff < -180] += 360
                unwrapped_lons = np.concatenate(([lons[0]], lons[0] + np.cumsum(lon_diff)))

                # Create interpolation functions
                lat_interp = interpolate.interp1d(times, lats, kind='linear')
                lon_interp = interpolate.interp1d(times, unwrapped_lons, kind='linear')

                # Generate interpolated points
                interp_times = np.linspace(times.min(), times.max(), num_interpolation_points)
                interp_lats = lat_interp(interp_times)
                interp_lons = lon_interp(interp_times) % 360
                interp_lons = np.where(interp_lons > 180, interp_lons - 360, interp_lons)

                interpolated[sat] = {
                    'times': interp_times,
                    'lats': interp_lats,
                    'lons': interp_lons,
                    'actions': sat_data['data']['actions'],
                    'task_being_collected': sat_data['data']['task_being_collected'],
                    'task_reward': sat_data['data']['task_reward'],
                    'rewards': sat_data['data']['rewards']
                }

        return interpolated

    def populate_satellite_data(self, satellite, steps):
        for step in steps:
            if satellite in step:
                sat_step = step[satellite]
                self.satellites[satellite]['data']['times'].append(sat_step['time'])
                self.satellites[satellite]['data']['lats'].append(sat_step['latitude'])
                self.satellites[satellite]['data']['lons'].append(sat_step['longitude'])
                self.satellites[satellite]['data']['actions'].append(sat_step['actions'])
                self.satellites[satellite]['data']['task_being_collected'].append(sat_step['task_being_collected'])
                self.satellites[satellite]['data']['rewards'].append(sat_step['reward'])
                self.satellites[satellite]['data']['task_reward'].append(sat_step['task_reward'])

    def interpolate_satellite(self, sat_data, num_points):
        times, lats, lons = map(np.array, [sat_data['times'], sat_data['lats'], sat_data['lons']])
        unique_times, unique_indices = np.unique(times, return_index=True)
        
        if len(unique_times) > 1:
            lat_interp = interpolate.interp1d(unique_times, lats[unique_indices], kind='linear')
            lon_interp = interpolate.interp1d(unique_times, lons[unique_indices], kind='linear')
            
            interp_times = np.linspace(unique_times.min(), unique_times.max(), num_points)
            return {
                'times': interp_times,
                'lats': lat_interp(interp_times),
                'lons': lon_interp(interp_times),
                'actions': sat_data['actions'],
                'task_being_collected': sat_data['task_being_collected'],
                'rewards': sat_data['rewards']
            }
        return None

    def setup_visualization(self):
        markers = ['o', 's', '^']  # circle for 1, square for 2, triangle for 3+
        # self.target_scatters = [
        #     self.ax.scatter([], [], transform=ccrs.Geodetic(), color='black', s=10, zorder=3, 
        #                     marker=markers[min(collects-1, 2)], label=f'{collects} Collect{"s" if collects > 1 else ""}')
        #     for collects in range(1, 4)
        # ]
        self.targets.reset_scatters()
        self.satellite_scatters = {sat: self.ax.scatter([], [], transform=ccrs.Geodetic(), 
                                                        color=data['color'], s=80, zorder=4) 
                                   for sat, data in self.satellites.items()}
        self.imaging_boxes = PathCollection([], facecolors='yellow', edgecolors='red', 
                                            alpha=0.3, linewidths=2, transform=ccrs.PlateCarree())
        self.ax.add_collection(self.imaging_boxes)
        self.reward_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                        verticalalignment='top', fontweight='bold', fontsize=14)
        plt.title('Target Locations and Satellite Positions')
        plt.legend()

    def draw_collection_lines(self, frame):
        # Remove old lines
        for line in self.collection_lines:
            line.remove()
        self.collection_lines.clear()

        for sat, sat_data in self.interpolated_satellites.items():
            step_index = int(frame * (len(sat_data['actions']) - 1) / (len(sat_data['times']) - 1))
            action = sat_data['actions'][step_index]
            if 0 <= action <= 9:
                task = sat_data['task_being_collected'][step_index]
                if task:
                    sat_lon, sat_lat = sat_data['lons'][frame], sat_data['lats'][frame]
                    target_lon, target_lat = task['longitude'], task['latitude']
                    line, = self.ax.plot([sat_lon, target_lon], [sat_lat, target_lat], 
                                        color='red', linestyle=':', transform=ccrs.Geodetic())
                    self.collection_lines.append(line)

        return self.collection_lines

    def update(self, frame):
        self.imaging_boxes.set_paths([])
        total_reward = 0

        self.targets.render()

        for sat, scatter in self.satellite_scatters.items():
            if sat in self.interpolated_satellites:
                sat_data = self.interpolated_satellites[sat]
                scatter.set_offsets([[sat_data['lons'][frame], sat_data['lats'][frame]]])
                
                step_index = int(frame * (len(sat_data['actions']) - 1) / (len(sat_data['times']) - 1))
                action = sat_data['actions'][step_index]
                
                if step_index < len(sat_data['rewards']):
                    total_reward += sat_data['rewards'][step_index]
                
                if 0 <= action <= 9:
                    task = sat_data['task_being_collected'][step_index]
                    if task:
                        self.targets.get_target(task['id']).set_collecting(sat_data['task_reward'][step_index])
                        self.add_imaging_box(task['longitude'], task['latitude'])

        self.reward_text.set_text(f"Score {total_reward:.2f}")

        artists = (list(self.satellite_scatters.values()) + 
               [self.imaging_boxes] + 
               self.targets.get_artists() +
               [self.reward_text])

        if self.draw_lines:
            artists += self.draw_collection_lines(frame)

        return artists

    def add_imaging_box(self, lon, lat):
        box_size = 10
        verts = [
            (lon - box_size/2, lat - box_size/2),
            (lon + box_size/2, lat - box_size/2),
            (lon + box_size/2, lat + box_size/2),
            (lon - box_size/2, lat + box_size/2),
            (lon - box_size/2, lat - box_size/2),
        ]
        path = Path(verts, [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY])
        self.imaging_boxes.set_paths(self.imaging_boxes.get_paths() + [path])

    # def update_targets(self, current_targets_being_collected):
        
    #     for i, scatter in enumerate(self.target_scatters):
    #         scatter.set_offsets(updated_targets[i])
    #         scatter.set_facecolors(updated_colors[i])

    def create_animation(self):
        num_frames = len(next(iter(self.interpolated_satellites.values()))['times'])
        return FuncAnimation(self.fig, self.update, frames=num_frames, interval=50, blit=True)

    def save_animation(self, filename='satellite_animation.mp4'):
        num_frames = len(next(iter(self.interpolated_satellites.values()))['times'])
        fps = 20
        duration = num_frames / fps
        
        print(f"Rendering animation: {num_frames} frames, {duration:.2f} seconds at {fps} fps")
        
        anim = self.create_animation()
        
        start_time = time.time()
        with tqdm(total=num_frames, desc="Generating animation", unit="frame") as pbar:
            anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'],
                      progress_callback=lambda i, n: self.update_progress(i, n, start_time, pbar))
        
        total_time = time.time() - start_time
        print(f"\nAnimation saved as '{filename}'")
        print(f"Total rendering time: {total_time:.2f} seconds")
        print(f"Average rendering speed: {num_frames / total_time:.2f} frames/second")

    def update_progress(self, current_frame, total_frames, start_time, pbar):
        pbar.update(1)
        elapsed_time = time.time() - start_time
        fps = current_frame / elapsed_time if elapsed_time > 0 else 0
        remaining_frames = total_frames - current_frame
        eta = remaining_frames / fps if fps > 0 else 0
        
        pbar.set_postfix({
            'FPS': f"{fps:.2f}",
            'ETA': f"{eta:.2f}s",
            'Elapsed': f"{elapsed_time:.2f}s"
        })

    def show_animation(self):
        anim = self.create_animation()
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Satellite Visualization Tool")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("-o", "--output", help="Path to the output MP4 file (default: satellite_animation.mp4)", 
                        default="satellite_animation_v3.mp4")
    parser.add_argument("-s", "--show", action="store_true", help="Show the animation live instead of saving to file")
    parser.add_argument("--draw-lines", action="store_true", help="Draw dotted lines from satellites to targets during collection")
    parser.add_argument("--interpolation-points", type=int, default=10, help="Number of interpolation points to use for satellite positions")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    vis = SatelliteVisualization(args.input_file, draw_lines=args.draw_lines, interpolation_points=args.interpolation_points)
    
    if args.show:
        vis.show_animation()
    else:
        vis.save_animation(args.output)

if __name__ == "__main__":
    main()