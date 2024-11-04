import eventlet  # Move this import to the top
eventlet.monkey_patch()  # Call monkey_patch before any other imports

import sys
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import argparse
import numpy as np

from rl.config import load_config
from agent import Agent

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
        CORS(self.app)  # Enable CORS
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.setup_routes()
        self.setup_socketio_events()

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
            step_info = self.agent.take_step()
            interpolated_sat_positions = step_info.get_interpolated_sat_positions()
            current_sat_state = step_info.get_current_sat_state()
            # Emit step information to connected clients
            # self.socketio.emit('step_update', convert_numpy_types(info))
            # return jsonify(convert_numpy_types(info))
            # Return status 200 to indicate success
            print(convert_numpy_types(interpolated_sat_positions))
            self.socketio.emit('step_update', {
                'interpolated_sat_positions': convert_numpy_types(interpolated_sat_positions),
                'current_sat_state': convert_numpy_types(current_sat_state)
            })

            current_acts_obs = step_info.get_current_actions_and_observations()
            
            self.socketio.emit('current_acts_obs', convert_numpy_types(current_acts_obs))

            task_info = self.agent.get_task_info()
            self.socketio.emit('tasks', convert_numpy_types(task_info))

            return jsonify({"message": "Step taken successfully."}), 200
        
        @self.app.route('/api/get_sat_info', methods=['GET'])
        def get_sat_info():
            sat_info = self.agent.get_sat_info()
            print(sat_info)
            return jsonify(convert_numpy_types(sat_info))
        
        @self.app.route('/api/current_step', methods=['GET'])
        def get_current_step():
            current_step = self.agent.get_info()
            return jsonify(convert_numpy_types(current_step))
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset_agent():
            try:
                self.agent = Agent(self.config, self.model_name, greedy=False)
                # Emit reset event to clients
                self.socketio.emit('agent_reset', {"message": "Agent has been reset successfully."})
                return jsonify({"message": "Agent has been reset successfully."}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500

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
            

    def run(self, debug=True, host='0.0.0.0', port=5000):
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
