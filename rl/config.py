
import argparse
import yaml

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
        else:
            return config_data
        
def parse_args():

    p = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, required=True,
                    help='the configuration file')
    p.add_argument('--name', type=str, default='v1',
                    help='the name of the run')
    args = p.parse_args()
    return args

