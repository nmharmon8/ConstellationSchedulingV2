import eventlet  # Move this import to the top
eventlet.monkey_patch()  # Call monkey_patch before any other imports

import sys
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import argparse
import numpy as np
import json
import uuid

from tqdm import tqdm

import requests

from rl.config import load_config
from agent import Agent

import time

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class App:
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
        self.agent = Agent(config, model_name, greedy=False)
        self.app = Flask(__name__)
        
        # Add CORS headers to all responses
        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response
        
        # Initialize CORS for the entire app
        CORS(self.app, resources={
            r"/*": {
                "origins": "http://localhost:3000",
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True
            }
        })
        
        # Initialize SocketIO with CORS settings
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins=["http://localhost:3000"],
            ping_timeout=60,
            ping_interval=25,
            async_mode='eventlet'
        )
        
        self.setup_routes()
        self.setup_socketio_events()
        
        # Initialize run control variables
        self.run_active = False
        self.run_task = None

    def take_step(self):
        step_info = self.agent.take_step()

        return {
            'interpolated_sat_positions': convert_numpy_types(step_info.get_interpolated_sat_positions()),
            'current_sat_state': convert_numpy_types(step_info.get_current_sat_state()),
            'current_acts_obs': convert_numpy_types(step_info.get_current_actions_and_observations()),
            'tasks': convert_numpy_types(self.agent.get_task_info()),
            'completed_tasks': convert_numpy_types(self.agent.get_completed_tasks())
        }
    
    def create_new_task(self, name, lat, lon, priority, task_type, min_elev, duration):
         
        name += "_" + uuid.uuid4().hex

        lat = float(lat)
        lon = float(lon)

        try:
            priority = float(priority)
        except:
            priority = 0.5

        if priority < 0 or priority > 1:
            priority = 0.5

        try:
            task_type = int(task_type)
        except:
            task_type = 0

        if task_type not in [0, 1]:
            task_type = 0

        try:
            min_elev = float(min_elev)
        except:
            min_elev = np.radians(45)

        try:
            duration = float(duration)
        except:
            duration = 60.0

        if duration < 0 or duration > 200:
            duration = 60.0
        
        # Create the task using the agent
        self.agent.add_new_task(
            name=name,
            lat=lat,
            lon=lon,
            priority=priority,
            task_type=task_type,
            min_elev=min_elev,
            duration=duration,
            user_id='taskgpt'
        )

    def setup_routes(self):

        @self.app.route('/api/data', methods=['GET'])
        def get_data():
            return jsonify({"message": "Hello from Flask backend!"})
        
        @self.app.route('/api/tasks', methods=['GET'])
        def get_tasks():
            task_info = self.agent.get_task_info()
            return jsonify(convert_numpy_types(task_info))
        
        @self.app.route('/api/observation/inspector', methods=['GET'])
        def get_observation_inspector():
            observation_inspector = self.agent.get_inspector_observation_state()
            return jsonify(convert_numpy_types(observation_inspector))
        
        @self.app.route('/api/step', methods=['POST'])
        def take_step():
            step_update = self.take_step()
            self.socketio.emit('step_update', convert_numpy_types(step_update))
            return jsonify({"message": "Step taken successfully."}), 200
        
    
        @self.app.route('/api/initial_state', methods=['GET'])
        def get_initial_state():
            step_info = self.agent.get_info()
            interpolated_sat_positions = step_info.get_interpolated_sat_positions()
            current_sat_state = step_info.get_current_sat_state()

            current_acts_obs = step_info.get_current_actions_and_observations()
            task_info = self.agent.get_task_info()
            
            initial_state = {       
                'interpolated_sat_positions': convert_numpy_types(interpolated_sat_positions),
                'current_sat_state': convert_numpy_types(current_sat_state),
                'current_acts_obs': convert_numpy_types(current_acts_obs),
                'tasks': convert_numpy_types(task_info),
                'completed_tasks': convert_numpy_types(self.agent.get_completed_tasks())
            }
            
            self.socketio.emit('step_update', convert_numpy_types(initial_state)) 
            return jsonify({"message": "Initial state sent successfully."}), 200

        
        # @self.app.route('/api/get_sat_info', methods=['GET'])
        # def get_sat_info():
        #     sat_info = self.agent.get_sat_info()
        #     return jsonify(convert_numpy_types(sat_info))
        
        # @self.app.route('/api/current_step', methods=['GET'])
        # def get_current_step():
        #     current_step = self.agent.get_info()
        #     return jsonify(convert_numpy_types(current_step))
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset_agent():
            try:
                # Stop any running thread
                if self.run_active:
                    self.run_active = False
                    # Give the run thread time to stop
                    eventlet.sleep(0.1)
                
                self.agent = Agent(self.config, self.model_name, greedy=False)
                
                # Get initial task info after reset
                task_info = self.agent.get_task_info()
                
                # Emit reset event and tasks to clients
                self.socketio.emit('agent_reset', {
                    "message": "Agent has been reset successfully.",
                    "tasks": convert_numpy_types(task_info)
                })
                
                return jsonify({"message": "Agent has been reset successfully."}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # New Run Route
        @self.app.route('/api/run/<int:time_ms>', methods=['POST'])
        def run(time_ms):
            if self.run_active:
                return jsonify({"message": "Run already in progress"}), 200
            self.run_active = True
            self.run_task = self.socketio.start_background_task(self.run_steps, time_ms)
            return jsonify({"message": f"Run started with interval {time_ms} ms"}), 200

        # New Pause Route
        @self.app.route('/api/pause', methods=['POST'])
        def pause():
            if self.run_active:
                self.run_active = False
                return jsonify({"message": "Run paused"}), 200
            return jsonify({"message": "Run not active"}), 200

        @self.app.route('/api/create_task', methods=['POST'])
        def create_new_task():
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['name','lat', 'lon', 'priority', 'task_type', 'min_elev', 'duration']
                if not all(field in data for field in required_fields):
                    return jsonify({"error": "Missing required fields"}), 400
                
                # Validate field types and ranges
                if not isinstance(data['task_type'], int) or data['task_type'] not in [0, 1]:
                    return jsonify({"error": "task_type must be 0 (rf) or 1 (imaging)"}), 400
                    
                if not 0 <= data['min_elev'] <= 90:
                    return jsonify({"error": "min_elev must be between 0 and 90 degrees"}), 400
                    
                if not 0 <= data['duration'] <= 200:
                    return jsonify({"error": "duration must be between 0 and 200 seconds"}), 400
                
                # Create the task using the agent
                new_task = self.agent.add_new_task(
                    name=data['name'],
                    lat=float(data['lat']),
                    lon=float(data['lon']),
                    priority=float(data['priority']),
                    task_type=int(data['task_type']),
                    min_elev=float(data['min_elev']),
                    duration=float(data['duration'])
                )
                
                # Get updated task info
                task_info = self.agent.get_task_info()
                
                # Emit task update to all connected clients
                self.socketio.emit('tasks_updated', convert_numpy_types(task_info))
                
                return jsonify({
                    "message": "Task created successfully",
                    "task": convert_numpy_types(new_task),
                    "tasks": convert_numpy_types(task_info)
                }), 201
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/generate', methods=['POST'])
        def generate_task():
            url = "http://localhost:5983/api/generate"
            headers = {"Content-Type": "application/json"}

            try:
                data = request.get_json()
               
                if 'prompt' not in data:
                    return jsonify({"error": "Missing prompt in request"}), 400

                prompt_data = {"prompt":data['prompt']}
                print("Prompt data: ", prompt_data)

                response = requests.post(url, json=prompt_data, headers=headers, stream=True)


                output_data = ""
                if response.status_code == 200:
                    # Process the response
                    for line in tqdm(response.iter_lines(decode_unicode=True)):
                        d = json.loads(line)
                        status = d.get('status', '')
                        if status == "info":
                            print("Info: ", d)
                            if "context" in d:
                                self.socketio.emit('task_gpt_info', {"message": "\n --- Context Retrieved --- \n" + d['context'], "done": False})
                            if "api_format" in d:
                                self.socketio.emit('task_gpt_info', {"message": "\n --- API Format --- \n" + d['api_format'], "done": False})

                        elif status == "llm_output":
                            print("LLM Output: ", d)
                            chunk = d.get('chunk', '')
                            self.socketio.emit('task_gpt_info', {"message": chunk.strip(), "done": True})

                        elif status == "final_output":
                            output_data = d.get('data', '')
                            print("Final Output: ", output_data)
                            break

                    for task in output_data:
                        self.create_new_task(task['name'], task['lat'], task['lon'], task['priority'], task['task_type'], task['min_elev'], task['duration'])


                    task_info = self.agent.get_task_info()
                    self.socketio.emit('tasks_updated', convert_numpy_types(task_info))

                    print("Tasks generated successfully")
                    
                    return jsonify({"message": "Tasks generated successfully"}), 200

            except Exception as e:
                print("Error: ", e)
                return jsonify({"error": str(e)}), 500

    def run_steps(self, time_ms):
        while self.run_active:
            start_time = time.time()
            step_update = self.take_step()
            self.socketio.emit('step_update', convert_numpy_types(step_update))
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000
            sleep_time_ms = time_ms - elapsed_time_ms
            if sleep_time_ms > 0:
                eventlet.sleep(sleep_time_ms / 1000)
            else:
                eventlet.sleep(0)  # Proceed immediately if the step took longer than the interval

    def setup_socketio_events(self):
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            emit('message', {'data': 'Connected to Flask-SocketIO server!'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')

        @self.socketio.on('custom_event')
        def handle_custom_event(json):
            print('Received custom event: ' + str(json))
            emit('response_event', {'data': 'Server received custom_event'}, broadcast=True)

    def run(self, debug=True, host='localhost', port=4000):
        # Use socketio.run instead of app.run
        self.socketio.run(self.app, debug=debug, host=host, port=port)

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, required=True,
                        help='the configuration file')
    parser.add_argument('--model', type=str, default="v9",
                        help='the model to load')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    app_instance = App(config, args.model)
    app_instance.run(debug=True)

    



"""
export PYTHONPATH=$PYTHONPATH:../../
python app.py --config ../../rl/configs/basic_config.yaml --model v59_full_fsw
"""
