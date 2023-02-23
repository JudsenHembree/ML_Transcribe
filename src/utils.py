"""
Utility functions for the project
"""
import sys
import json
import os
import wave
import numpy as np
from matplotlib import pyplot as plt
from subprocess import run, PIPE, CalledProcessError
from glob import glob

def reconfigure():
    """Reconfigures the config file"""
    config = get_config()
    pwd = os.getcwd()
    print("Current data home is %s", config["data_home"])
    config["data_home"] = pwd + "/data"
    print("New data home is %s", config["data_home"])
    print("Current model directory is %s", config["MODEL_DIRECTORY"])
    config["MODEL_DIRECTORY"] = pwd + "/models"
    print("New model directory is %s", config["MODEL_DIRECTORY"])

    with open('/ML_Transcribe/config/config.json', 'w') as conf:
        json.dump(config, conf)

def check_data_home(data_home):
    """Checks if the data home has two folders"""
    if len(os.listdir(data_home)) > 2:
        print("Error: Data home has more than two folders")
        print("Folders should be archive and the active folder")
        sys.exit(2)

def get_active_folder(data_home):
    """Returns the active folder"""
    check_data_home(data_home)
    for folder in glob(data_home + "/*"):
        if os.path.isdir(folder):
            if not folder.endswith("archive"):
                print("Active folder is %s", folder)
                return folder
    return None

def attempt_archive(data_home):
    """Attempts to archive prior run into the archive folder"""
    dat = glob(data_home + "/*")
    print(dat)
    if "archive" not in os.listdir(data_home):
        print("Creating archive folder")
        os.makedirs(os.path.join(data_home, "archive"))
    for folder in dat:
        if os.path.isdir(folder):
            if not folder.endswith("archive"):
                try:
                    path = os.path.join(data_home, "archive")
                    print("Archiving %s to %s", folder, path)
                    proc = run(["mv", folder, path], \
                            stdout=PIPE, stderr=PIPE, check=True)
                    if proc.returncode != 0:
                        print("Error: %s", str(proc.stderr))
                except CalledProcessError as err:
                    print("Error: %s", str(err))

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

def graph_all_wav_for_each_folder(folder):
    """Graphs all .wav files in a folder Intended to be used on the seperated folder"""
    # find seperated folder
    for file in os.listdir(folder):
        if file.endswith("seperated"):
            for innerFile in os.listdir(os.path.join(folder, file)):
                if os.path.isdir(os.path.join(folder, file, innerFile)):
                    graph_all_wav(os.path.join(folder, file, innerFile))

def graph_all_wav(folder):
    """Graphs all .wav files in a folder"""
    for file in os.listdir(folder):
        print("Graphing %s", file)
        if file.endswith(".wav"):
            graph_wav(os.path.join(folder, file))

def graph_wav(file):
    """Graphs a .wav file"""
    try:
        wav = wave.open(file, 'r')
    except FileNotFoundError:
        print("Error: %s not found", file)
        return
    frames = wav.readframes(-1)
    sound_info = np.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    try:
        fig = plt.figure()
        plt.title(file)
        plt.plot(sound_info)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.savefig(file + ".png")
        plt.close(fig)
    except ValueError:
        print("Error: %s is empty", file)
    except MemoryError:
        print("Error: %s is too large", file)
    except Exception as err:
        print("Error: %s", str(err))

def cull_data_home(data_home):
    """Deletes the data home"""
    try:
        proc = run(["rm", "-rf", data_home], stdout=PIPE, stderr=PIPE, check=True)
        if proc.returncode != 0:
            print("Error: %s", str(proc.stderr))
    except CalledProcessError as err:
        print("Error: %s", str(err))
    make_data_home(data_home)

def make_data_home(data_home):
    """Makes the data home"""
    if not os.path.exists(data_home):
        os.makedirs(data_home)
