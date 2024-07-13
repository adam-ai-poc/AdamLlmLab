import yaml
import time
import functools
import os

# DEBUG = True if os.environ.get("MODE") == DEBUG

# Read specific config
def read_config(file_path, key):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except:
            print("Error reading config file")
    return config[key]

# Read all config in the given file
def read_all_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except:
            print("Error reading config file")
    return config


def benchmark(func):
    """Decorator that prints the time a function takes to execute."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"--- Function '{func.__name__}' took {end_time - start_time:.4f} seconds to execute. ---")
        return result
    return wrapper

# Recursively search for the persist_directory key
def find_persist_directory(config):
    if isinstance(config, dict):
        for key, value in config.items():
            if key == "persist_directory":
                return value
            elif isinstance(value, dict):
                result = find_persist_directory(value)
                if result:
                    return result
    return None

# Check if persist_directory exists
def check_persist_directory(config):
    persist_directory = find_persist_directory(config)
    if persist_directory:
        if os.path.exists(persist_directory):
            return True
        else:
            return False
    else:
        return False