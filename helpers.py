import os
import json
import pickle
import pandas as pd
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
    """
    loads .json file located at location `path_to_file`
    and returns a dictionary.
    """
    with open(path_to_file, "r") as f:
        data = json.load(f)
    return data

data_types_ = Literal["train", "dev"] # type of dataset `train`, `test` or `validation (devset)`
extract_forms_ = Literal["judgement", "preamble", "combined"] # type of task data `judgement` or `preamble`

def read_data(path_to_data: str,
              data_type: data_types_ = "train", 
              extract_form: extract_forms_ = "judgement"):

    """
    utilizes data_type; either [`train`, `dev`] and extract_form;`
    either [`judgement`, `preamble`] to load dataset stored as a 
    .json file.
    """
    path_to_data_file = os.path.join(os.path.join(path_to_data, f"./NER_{data_type.upper()}"), f"NER_{data_type.upper()}_{extract_form.upper()}.json")
    data = load_json_file(path_to_data_file)
    return data

def save_to_pickle(path_to_file, itemlist):
    """ 
    saves a python list to .pkl file at location 
    specified using `path_to_file`
    NOTE extenson -> .pkl (or `pickle`)
    """
    with open(path_to_file, 'wb') as fp:
        pickle.dump(itemlist, fp)
    print(f"saved file as pickle @ location: {path_to_file}")
 
def load_from_pickle(path_to_file):
    """
    loads a .pkl file as python list stored at
    location `path_to_file`
    """
    with open (path_to_file, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist

def save_to_json(path_to_file: str, data: dict):
    """
    save a dictionary to .json file at location 
    specified via `path_to_file`
    """
    with open(path_to_file, 'w') as outfile:
        json.dump(data, outfile)
    print(f"Files saved at location: {path_to_file}")

def save_results(path_to_file: str, results: dict):
    """
    save NER results calculated using `seqeval` metric
    to .csv file by disregarding non-important 
    overall prediction scores. 
    """
    classnames, f1_scores, precision_scores, recall_scores, numbers = [], [], [], [], []
    for key, value in results.items():
        if not key.startswith("overall"):
            classnames.append(key)
            f1_scores.append(results[key]["f1"])
            precision_scores.append(results[key]["precision"])
            recall_scores.append(results[key]["recall"])
            numbers.append(results[key]["number"])

    results_df = pd.DataFrame({
        "CLASS": classnames,
        "F1": f1_scores,
        "PRECISION": precision_scores,
        "RECALL": recall_scores,
        "COUNT": numbers
    })
    results_df.set_index("CLASS", inplace=True)
    results_df.to_csv(path_to_file)
    print(f"File saved at location: {path_to_file}")

def load_config(path_to_yaml_file: str):
    """
    load the configuration file as .yaml stored 
    (and edited) at location `path_to_file`.
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

# prerequisites
config_data["PATH_TO_RESULT_OUTPUT_DIR"] = os.path.join(config_data["PATH_TO_RESULT_OUTPUT_DIR"], "./{}_{}".format(config_data['MODEL_CHECKPOINT'], config_data['VERSION']))
config_data["PATH_TO_MODEL_OUTPUT_DIR"] = os.path.join(config_data["PATH_TO_MODEL_OUTPUT_DIR"], "./{}_{}".format(config_data['MODEL_CHECKPOINT'], config_data['VERSION']))
check_and_create_directory(config_data["PATH_TO_RESULT_OUTPUT_DIR"])
check_and_create_directory(config_data["PATH_TO_MODEL_OUTPUT_DIR"])
check_and_create_directory(config_data["PATH_TO_RESULT_OUTPUT_DIR"] + "./confusion_matrix") # separate directory for confusion matrix
check_and_create_directory(config_data["PATH_TO_RESULT_OUTPUT_DIR"] + "./final_results_dev") # separate directory for dev-set results
