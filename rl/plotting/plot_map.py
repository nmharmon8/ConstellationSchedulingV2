import json
from collections import defaultdict
# Set matplotlib to use interactive backend
import matplotlib
matplotlib.use('TkAgg')
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
import cProfile
import pstats
import io
from matplotlib.patches import Rectangle

from rl.plotting.object_def import ObjectDef


class TaskScatter:

    def __init__(self, task):
        self.task = task
        self.markers = {
            'RF': ['o', 's', '^'],
            'IMAGING': ['o', 's', '^'],
            'DATA_DOWNLINK': ['d']
        }

        self.colors = {
            'RF': 'black',
            'IMAGING': 'black',
            'DATA_DOWNLINK': 'purple'
        }

        self.sizes = {
            'RF': 10,
            'IMAGING': 10,
            'DATA_DOWNLINK': 40
        }
        
        self.reset()

    def reset(self):
        self.marker = self.markers[self.task.tasking_type][max(0, self.task.simultaneous_collects_required - 1)]
        self.color = self.colors[self.task.tasking_type]
        self.size = self.sizes[self.task.tasking_type]

    @property
    def lat(self):
        return self.task.lat
    
    @property
    def lon(self):
        return self.task.lon
        
    def set_collecting(self, task_reward):
        if self.task.tasking_type != 'DATA_DOWNLINK':
            self.color = 'red'
            if task_reward> 0:
                self.color = 'green'
            


class ScatterTasks:
    def __init__(self, ax, tasks):
        self.tasks = {}
        self.ax = ax
        self.scatters = {}
        self.tasks = {task.id: TaskScatter(task) for task in tasks}

    def reset_scatters(self):
        markers = set([task.marker for task in self.tasks.values()])
        # TODO: Add label to the scatter
        self.scatters = {m: self.ax.scatter([], [], transform=ccrs.Geodetic(), s=10, zorder=4, marker=m) for m in  markers}

        if len(self.tasks) > 0:
            for marker in markers:
                # Expects lon, lat where lon is x and lat is y
                self.scatters[marker].set_offsets([(task.lon, task.lat) for task in self.tasks.values() if task.marker == marker])
                self.scatters[marker].set_facecolors([task.color for task in self.tasks.values() if task.marker == marker])
                self.scatters[marker].set_sizes([task.size for task in self.tasks.values() if task.marker == marker])

    def render(self):
        self.reset_scatters()
        
    def get_task(self, task_id):
        return self.tasks[task_id]
    
    def get_tasks(self):
        return self.tasks.values()

    def reset(self):
        for task in self.tasks.values():
            task.reset()

    def get_artists(self):
        return list(self.scatters.values())

class SatelliteVisualization:
    def __init__(self, data_file, draw_lines=False, interpolation_points=50):
        self.object_def = ObjectDef(data_file, interpolation_points)
        self.fig, self.ax = self.setup_map()
        self.sat_to_color = self.setup_sat_dots()
        self.tasks = ScatterTasks(self.ax, self.object_def.tasks)
        self.draw_lines = draw_lines
        self.collection_lines = []
        self.storage_bars = {}
        self.storage_fill_bars = {}  # New attribute
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

    def setup_sat_dots(self):
        satellite_ids = self.object_def.get_satellite_ids()
        colors = plt.cm.viridis(np.linspace(0, 1, len(satellite_ids)))
        sat_to_color = {sat: colors[i] for i, sat in enumerate(satellite_ids)}
        return sat_to_color
    
    def setup_visualization(self):
        self.tasks.reset_scatters()
        self.satellite_scatters = {sat: self.ax.scatter([], [], transform=ccrs.Geodetic(), 
                                                        color=color, s=80, zorder=4) 
                                   for sat, color in self.sat_to_color.items()}
        self.imaging_boxes = PathCollection([], facecolors='yellow', edgecolors='red', 
                                            alpha=0.3, linewidths=2, transform=ccrs.PlateCarree())
        self.ax.add_collection(self.imaging_boxes)
        self.reward_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                        verticalalignment='top', fontweight='bold', fontsize=14)
        plt.title('Target Locations and Satellite Positions')
        plt.legend()
        self.setup_storage_bars()

    def setup_storage_bars(self):
        for sat_id in self.object_def.get_satellite_ids():
            bar = self.ax.add_patch(Rectangle((0, 0), 5, 5, facecolor='lightgray', edgecolor='black', alpha=0.7))
            self.storage_bars[sat_id] = bar

    def draw_collection_lines(self, frame):
        # Remove old lines
        for line in self.collection_lines:
            line.remove()
        self.collection_lines.clear()

        for sat_id in self.object_def.get_satellite_ids():
            task = self.object_def.get_task_being_collected(sat_id, frame)
            #TODO add special line type for downlink
            if task.is_collect_tasking or task.is_downlink_tasking:
                sat_lon, sat_lat = self.object_def.get_sat_lon_lat(sat_id, frame)
                line, = self.ax.plot([sat_lon, task.lon], [sat_lat, task.lat], 
                                    color='red', linestyle=':', transform=ccrs.Geodetic())
                self.collection_lines.append(line)

        return self.collection_lines

    def update(self, frame):

        start = time.time()

        self.imaging_boxes.set_paths([])
        self.tasks.render()

        cum_reward = 0
        task_completed = 0
        step = 0

        for sat, scatter in self.satellite_scatters.items():
            sat_lon, sat_lat = self.object_def.get_sat_lon_lat(sat, frame)
            scatter.set_offsets([sat_lon, sat_lat])
            cum_reward = self.object_def.get_cum_reward(frame)
            task_completed = self.object_def.get_n_tasks_collected(frame)
            step = self.object_def.get_step(frame)
            task = self.object_def.get_task_being_collected(sat, frame)
            if task.is_collect_tasking or task.is_downlink_tasking:
                self.tasks.get_task(task.id).set_collecting(task.task_reward)
                self.add_imaging_box(task.lon, task.lat)
            
            # Update storage bar
            storage_pct = self.object_def.get_storage_percentage(sat, frame)
            self.update_storage_bar(sat, sat_lon, sat_lat, storage_pct)

        # self.reward_text.set_text(f"Score {cum_reward:.2f} Task Completed: {task_completed} Step: {step}")

        artists = (list(self.satellite_scatters.values()) + 
               [self.imaging_boxes] + 
               self.tasks.get_artists() +
               [self.reward_text] +
               list(self.storage_bars.values()) +
               list(self.storage_fill_bars.values()))

        if self.draw_lines:
            artists += self.draw_collection_lines(frame)

        stop = time.time()
        print(f"Update time: {stop - start:.2f} seconds")

        return artists

    def update_storage_bar(self, sat_id, lon, lat, storage_pct):

        bar = self.storage_bars[sat_id]
        bar_width = 10  # degrees
        bar_height = 1  # degrees
        
        # Position the bar directly below the satellite dot
        bar_x = lon - bar_width / 2
        bar_y = lat - bar_height - 3.0  # Offset by bar height plus a small gap
        
        bar.set_xy((bar_x, bar_y))
        bar.set_width(bar_width)  # Set full width
        bar.set_height(bar_height)
        
        # Create a two-color rectangle
        bar.set_facecolor('lightgray')  # Background color
        bar.set_edgecolor('black')
        
        # Add a colored rectangle representing the storage percentage
        filled_width = bar_width * storage_pct
        filled_rect = Rectangle((bar_x, bar_y), filled_width, bar_height, 
                                facecolor='blue', edgecolor='none')
        self.ax.add_patch(filled_rect)
        self.storage_fill_bars[sat_id] = filled_rect


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

    def create_animation(self):
        num_frames = self.object_def.get_number_of_frames()
        return FuncAnimation(self.fig, self.update, frames=num_frames, interval=50, blit=True)

    def save_animation(self, filename='satellite_animation.mp4'):
        num_frames = self.object_def.get_number_of_frames()
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
    parser.add_argument("--draw-lines", action="store_true", help="Draw dotted lines from satellites to tasks during collection")
    parser.add_argument("--interpolation-points", type=int, default=10, help="Number of interpolation points to use for satellite positions")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    vis = SatelliteVisualization(args.input_file, draw_lines=args.draw_lines, interpolation_points=args.interpolation_points)

    if args.show:
        vis.show_animation()
    else:
        vis.save_animation(args.output)

    if args.profile:
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

if __name__ == "__main__":
    main()


"""
python -m rl.plotting.plot_map data.json  -s --draw-lines 
"""