#! /usr/bin/python
"""
This app is a music downloader that uses the spotdl library to download music from spotify
and the spleeter library to separate the music into its different parts
"""
import sys
import shutil
import torch
from pathlib import Path as path
import subprocess
from getopt import getopt, GetoptError
from glob import glob
from spotdl import Spotdl
import utils
import model
import data

def usage():
    """Prints the usage of the program"""
    print("Usage:\npython app.py [options]\n")
    print("Options:")
    print("\t-h --help: Show this help message and exit")
    print("\t-c --config: specify a config file, not advised")

def main():
    """Main function of the program"""

    GRAPH = False
    RECORD = False
    NEW = False
    CULL = False
    RECONF = False
    MIDI = False
    TRAIN = False
    LEGACY = False
    ML_Home = path("/ML_Transcribe")

    """Parse command line arguments"""
    try:
        opts, _ = getopt(sys.argv[1:], "rhgndcm:", ["legacy", "train", "midi", "reconfig", "help", "graph", "new", "delete", "record", "config="])
    except GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, _ in opts:
        if opt in ("-r", "--reconfig"):
            print("Reconfiguring.")
            RECONF = True
        elif opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-c", "--config"):
            print("alt Config file not currently supported")
            sys.exit(2)
        elif opt in ("-g", "--graph"):
            print("Generating graphs.")
            GRAPH = True
        elif opt in ("-n", "--new"):
            print("Creating a new session folder.")
            NEW = True
        elif opt in ("-d", "--delete"):
            print("Delete data in data_home folder")
            CULL = True
        elif opt in ("-m", "--midi"):
            print("Converting wav to midi")
            MIDI = True
        elif opt in ("--record"):
            print("Recording")
            RECORD = True
        elif opt in ("--train"):
            print("Training")
            TRAIN = True
        elif opt in ("--legacy"):
            print("Using legacy data")
            LEGACY = True
        else:
            print("Unknown option" + opt) 
            usage()
            sys.exit(2)

    if RECONF:
        ML_Home = utils.reconfigure()

    config, file = utils.get_config()

    if CULL:
        utils.cull_data_home(config["data_home"])
        utils.make_data_home(config["data_home"])

    if NEW:
        utils.attempt_archive(config["data_home"])
        print("Welcome to the music downloader!")
        print("Enter the url of a spotify playlist or song")
        url = input()

        print("Name the folder you want to save the music to")
        fold_end = input()


        session_folder, seperated_folder = utils.create_unique_folder(config["data_home"], fold_end)
        spotdl = Spotdl(config["client_id"], config["client_secret"])
        songs = spotdl.search([url])
        spotdl.download_songs(songs)

        songs_downloaded_path = path(session_folder + "/*.mp3")
        # get downlaoded and move them to session folder
        songs_downloaded = glob("*.mp3")
        for song in songs_downloaded:
            shutil.move(song, session_folder + "/" + song)

        songs_downloaded = glob(str(songs_downloaded_path))
        # separate the music into its different parts
        cmd = ["spleeter", "separate", "-o", seperated_folder, "-p", "spleeter:5stems"]
        for song in songs_downloaded:
            cmd.append(song)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as err:
                print("Error: %s", str(err))
            cmd.pop()

        # clean up low apmlitude on each wav
        utils.clean_wav(seperated_folder)
        # convert spotdl downloaded mp3 to wav to standardize all audio to wav
        utils.convert_all_mp3_to_wav(session_folder)
        # generate a selection for each wav file (10 seconds of highest amplitude)
        utils.generate_selections_for_each_folder(seperated_folder)

    if LEGACY:
        print("You are about to use pre-split audio files, and this will clean" \
                "the data_home folder. Are you sure you want to continue? (y/n)")
        answer = input()
        if answer == "y":
            utils.cull_data_home(config["data_home"])
            utils.make_data_home(config["data_home"])
            utils.legacy(config["data_home"])
        else:
            sys.exit(2)

    if RECORD:
        active_folder = utils.get_active_folder(config["data_home"])
        if active_folder is None:
            print("No active folder found")
            sys.exit(2)
        utils.record(active_folder)

    if TRAIN:
        active_folder = utils.get_active_folder(config["data_home"])
        utils.collect_recordings_place_in_folder(active_folder)
        metadata = utils.generate_meta_data(active_folder)
        # generate the melspectrograms for each wav file recording
        if GRAPH:
            utils.generate_CNN_inputs(active_folder)

        # init the dataset
        dataset = data.Data(metadata)
        items = len(dataset)
        train = int(items * 0.8)
        val = items - train
        # random split is how we split into training and test sets.
        train_set, val_set = torch.utils.data.random_split(dataset, [train, val])

        # create data loaders to load the data in batches
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

        # create the model
        # put on gpu if possible
        model_to_train = model.Model()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_to_train.to(device)

        # train model
        model.training(model_to_train, train_loader, 200, device)

if __name__ == "__main__":
    main()
