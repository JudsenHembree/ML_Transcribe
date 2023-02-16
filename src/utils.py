"""
Utility functions for the project
"""
import json
import os

def get_config(file='/ML_Transcribe/config/config.json'):
    """Reads the config file and returns a dictionary"""
    with open(file, encoding="utf-8") as conf:
        config = json.load(conf)
    return config

def create_unique_folder(data_home, fold_end):
    """Creates a unique folder for the session fold end is usually user input"""
    folder = os.path.join(data_home, fold_end, "mp3")
    seperated_folder = os.path.join(data_home, fold_end, "seperated")
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(seperated_folder):
        os.makedirs(seperated_folder)
    return folder, seperated_folder
