import os
import sys

def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_data_path():
    path = os.path.join(get_root_path(), 'data')
    os.makedirs(path, exist_ok=True)
    return path

def get_results_path():
    path = os.path.join(get_root_path(), 'results')
    os.makedirs(path, exist_ok=True)
    return path

def get_data_path():
    path = os.path.join(get_root_path(), 'data')
    os.makedirs(path, exist_ok=True)
    return path

if __name__ == '__main__':
    print(get_root_path())
    print(get_results_path())
