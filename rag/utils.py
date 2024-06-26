import yaml

def read_config(file_path, key):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except:
            print("Error reading config file")
    return config[key]