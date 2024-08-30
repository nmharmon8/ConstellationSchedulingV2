import json
import argparse


def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Decorator that prints "#####" before and after the function
def check_header(func):
    def wrapper(*args, **kwargs):
        print(f"##### STARTING CHECK {func.__name__} #####")
        func(*args, **kwargs)
        print(f"##### ENDING CHECK {func.__name__} #####\n")
    return wrapper

@check_header
def check_for_duplicate_task_collections_at_same_step(data):
    report_str = ""
    collected_tasks_by_step = {}
    for step in data['steps']:
        for sat_id, sat_data in step['satellites'].items():
            if sat_data['action_type']=='collect':
                task = sat_data['task_being_collected']
                if sat_data['task_reward'] > 0:
                    if task['id'] in collected_tasks_by_step:
                        report_str += f"\nDuplicate task collection: {task['id']} at step {step['step']} Orignally at step {collected_tasks_by_step[task['id']]}"
                    collected_tasks_by_step[task['id']] = step['step']

    if report_str != "":
        print(f"Found duplicate task collections at the same step:\n{report_str}")
    else:
        print("No duplicate task collections found at the same step")


@check_header
def check_for_duplicate_task_collections_at_different_steps(data):
    report_str = ""
    collected_tasks_by_step = {}
    for step in data['steps']: 
        for sat_id, sat_data in step['satellites'].items():
            if sat_data['action_type']=='collect':
                task = sat_data['task_being_collected']
                if sat_data['task_reward'] > 0:
                    if task['id'] in collected_tasks_by_step:
                        if collected_tasks_by_step[task['id']] != step['step']:
                            report_str += f"\nDuplicate task collection: {task['id']} at step {step['step']} Orignally at step {collected_tasks_by_step[task['id']]}"
                    collected_tasks_by_step[task['id']] = step['step']
    if report_str != "":
        print(f"Found duplicate task collections at different steps:\n{report_str}")
    else:
        print("No duplicate task collections found at different steps")

@check_header
def check_for_failed_collections(data):
    report_str = ""
    for step in data['steps']:
        for sat_id, sat_data in step['satellites'].items():
            if sat_data['action_type']=='collect':
                task = sat_data['task_being_collected']
                if task is None:
                    report_str += f"\nFailed collection: {sat_id} at step {step['step']} no task being collected"
                elif sat_data['task_reward'] <= 0:
                    report_str += f"\nFailed collection: {task['id']} at step {step['step']} required collections: {task['simultaneous_collects_required']}"

    if report_str != "":
        print(f"Found failed collections:\n{report_str}")
    else:
        print("No failed collections found")

@check_header
def check_cum_score(data):
    cum_reward = 0
    task_rewards_sum = 0
    for step in data['steps']:
        for sat_id, sat_data in step['satellites'].items():
            if sat_data['action_type']=='collect' and sat_data['task_reward'] > 0:
                cum_reward += sat_data['task_reward']
                task_rewards_sum += sat_data['task_reward']
    print(f"Cumulative reward: {cum_reward}")
    print(f"Cumulative task reward: {task_rewards_sum}")

@check_header
def count_tasks_completed(data):
    num_tasks_completed = 0
    for step in data['steps']:
        for sat_id, sat_data in step['satellites'].items():
            if sat_data['action_type']=='collect' and sat_data['task_reward'] > 0:
                num_tasks_completed += 1
    print(f"Number of tasks completed: {num_tasks_completed}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Satellite Visualization Tool")
    parser.add_argument("input_file", help="Path to the input JSON file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    data = load_data(args.input_file)
    check_for_duplicate_task_collections_at_same_step(data)
    check_for_duplicate_task_collections_at_different_steps(data)
    check_for_failed_collections(data)
    check_cum_score(data)
    count_tasks_completed(data)

if __name__ == "__main__":
    main()
