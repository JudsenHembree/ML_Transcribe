import json

def get_config(file='/ML_Transcribe/config/config.json'):
    with open(file) as f:
        config = json.load(f)
    return config

def create_unique_folder(data_home, fold_end):
    import os
    folder = os.path.join(data_home, fold_end, "mp3")
    seperated_folder = os.path.join(data_home, fold_end, "seperated")
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(seperated_folder):
        os.makedirs(seperated_folder)
    return folder, seperated_folder

