#! /usr/bin/python

from spleeter.separator import Separator
from spotdl import Spotdl
import sys
import subprocess
import utils
from getopt import getopt, GetoptError
from glob import glob

def usage():
    print("Usage:\npython app.py [options]\n")
    print("Options:")
    print("\t-h --help: Show this help message and exit")
    print("\t-c --config: specify a config file, not advised")

def main():
    try:
        opts, args = getopt(sys.argv[1:], "hc:", ["help", "config="])
    except GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-c", "--config"):
            print("alt Config file not currently supported")
            sys.exit(2)
        else:
            print("Unknown option")
            usage()
            sys.exit(2)
    config = utils.get_config()
    print("Welcome to the music downloader!")
    print("Enter the url of a spotify playlist or song")
    url = input()

    print("Name the folder you want to save the music to")
    fold_end = input()

    session_folder, seperated_folder = utils.create_unique_folder(config["data_home"], fold_end)
    spotdl = Spotdl(config["client_id"], config["client_secret"], output=session_folder)
    songs = spotdl.search([url])
    results = spotdl.download_songs(songs)

    songs_downloaded = glob(session_folder + "/*.mp3")
    cmd = ["spleeter", "separate", "-o", seperated_folder, "-p", "spleeter:5stems"]
    for song in songs_downloaded:
        cmd.append(song)
        try:
            proc = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("Error: {}".format(e))
        cmd.pop()

        

if __name__ == "__main__":
    main()
