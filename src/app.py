#! /usr/bin/python
"""
This app is a music downloader that uses the spotdl library to download music from spotify
and the spleeter library to separate the music into its different parts
"""
import sys
from pathlib import Path as path
import subprocess
from getopt import getopt, GetoptError
from glob import glob
from spotdl import Spotdl
import utils

def usage():
    """Prints the usage of the program"""
    print("Usage:\npython app.py [options]\n")
    print("Options:")
    print("\t-h --help: Show this help message and exit")
    print("\t-c --config: specify a config file, not advised")

def main():
    """Main function of the program"""

    GRAPH = False
    NEW = False
    CULL = False
    RECONF = False
    MIDI = False
    Custom_Data = False
    ML_Home = path("/ML_Transcribe")

    """Parse command line arguments"""
    try:
        opts, _ = getopt(sys.argv[1:], "rhgndcm:", ["Custom", "midi", "reconfig", "help", "graph", "new", "delete", "config="])
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
            print("Graphing all .wav files post split.")
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
        elif opt in ("--Custom"):
            print("Converting wav to custom data")
            Custom_Data = True
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
        spotdl = Spotdl(config["client_id"], config["client_secret"], output=session_folder)
        songs = spotdl.search([url])
        spotdl.download_songs(songs)

        songs_downloaded_path = path(session_folder + "/*.mp3")
        songs_downloaded = glob(str(songs_downloaded_path))
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

    if GRAPH:
        active_folder = utils.get_active_folder(config["data_home"])
        if active_folder is None:
            print("No active folder found")
            sys.exit(2)
        utils.graph_all_wav_for_each_folder(active_folder, MIDI)

    if Custom_Data:
        active_folder = utils.get_active_folder(config["data_home"])
        utils.remove_non_spleeter_for_Active_folder(active_folder)
        utils.custom_data_transform(active_folder)


if __name__ == "__main__":
    main()
