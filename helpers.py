import os
import json
import pickle
import yaml
from typing import Literal

def check_and_create_directory(path_to_folder):
    """
    check if a nested path exists and create 
    missing nodes/directories along the route
    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    return path_to_folder

def load_json_file(path_to_file):
    with open(path_to_file, "r") as f:
        data = json.load(f)
    return data

data_types_ = Literal["train", "dev"] # type of dataset `train`, `test` or `validation (devset)`
extract_forms_ = Literal["judgement", "preamble"] # type of task data `judgement` or `preamble`

def read_data(path_to_data: str,
              data_type: data_types_ = "train", 
              extract_form: extract_forms_ = "judgement"):

    path_to_data_file = os.path.join(os.path.join(path_to_data, f"./NER_{data_type.upper()}"), f"NER_{data_type.upper()}_{extract_form.upper()}.json")
    data = load_json_file(path_to_data_file)
    return data

def save_to_pickle(path_to_file, itemlist):
    """ extenson -> .pkl """
    with open(path_to_file, 'wb') as fp:
        pickle.dump(itemlist, fp)
    print(f"saved file as pickle @ location: {path_to_file}")

def load_from_pickle(path_to_file):
    with open (path_to_file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist

def load_config(path_to_yaml_file: str):
    """
    load the configuration file as .yaml
    """
    with open(path_to_yaml_file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            print("loaded config.yaml successfully!")
        except yaml.YAMLError as exc:
            print(exc)
            data = None
    return data

PATH_TO_CONFIG_FILE = "./config.yaml"
config_data = load_config(PATH_TO_CONFIG_FILE)