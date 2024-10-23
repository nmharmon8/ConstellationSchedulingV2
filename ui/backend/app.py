import sys
from flask import Flask, jsonify
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
        self.agent = Agent(config, model_name, greedy=False)
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/api/data', methods=['GET'])
        def get_data():
            return jsonify({"message": "Hello from Flask backend!"})
        
        @self.app.route('/api/tasks', methods=['GET'])
        def get_tasks():
            task_info = self.agent.get_task_info()
            print(task_info[0])
            return jsonify(convert_numpy_types(task_info))
        
        @self.app.route('/api/step', methods=['GET'])
        def get_step():
            step_info = self.agent.take_step()
            return jsonify(convert_numpy_types(step_info))
        
        @self.app.route('/api/get_sat_info', methods=['GET'])
        def get_sat_info():
            sat_info = self.agent.get_sat_info()
            print(sat_info)
            return jsonify(convert_numpy_types(sat_info))

    def run(self, debug=True, host='0.0.0.0', port=5000):
        self.app.run(debug=debug, host=host, port=port)

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
python app.py --config ../../rl/configs/basic_config.yaml --model v58_full_fsw
"""