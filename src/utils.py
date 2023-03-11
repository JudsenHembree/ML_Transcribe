"""
Utility functions for the project
"""
from pathlib import Path as path
from glob import glob
from shutil import rmtree
from scipy.io import wavfile
import scipy.signal as signal
import sys
import os
import json
import wave
from matplotlib import pyplot as plt
import numpy as np
import librosa
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from visual_midi import Plotter
from visual_midi import Preset

global MIDI

def reconfigure():
    """Reconfigures the config file to dodge symlink issues"""
    config, file = get_config()
    print(config)
    print(file)
    pwd = os.getcwd()
    print("Current data home is %s", str(config["data_home"]))
    config["data_home"] = str(path(pwd + "/data"))
    print("New data home is %s", str(config["data_home"]))
    print("Current model directory is %s", str(config["MODEL_DIRECTORY"]))
    config["MODEL_DIRECTORY"] = str(path(pwd + "/models"))
    print("New model directory is %s", str(config["MODEL_DIRECTORY"]))

    with open(file, 'w') as conf:
        print(config)
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
    glob_path = path(data_home + "/*")
    for folder in glob(str(glob_path)):
        if os.path.isdir(folder):
            if not folder.endswith("archive"):
                print("Active folder is %s", folder)
                return folder
    return None

def attempt_archive(data_home):
    """Attempts to archive prior run into the archive folder"""
    if not os.path.exists(data_home):
        print("Warning: %s not found", data_home)
        print("Creating data home")
        os.makedirs(data_home)
        return
    glob_path = path(data_home + "/*")
    dat = glob(str(glob_path))
    print(dat)
    if "archive" not in os.listdir(data_home):
        print("Creating archive folder")
        os.makedirs(os.path.join(data_home, "archive"))
    for folder in dat:
        if os.path.isdir(folder):
            if not folder.endswith("archive"):
                arcpath = os.path.join(data_home, "archive")
                print("Archiving %s to %s", folder, path)
                renamed = os.path.join(arcpath, os.path.basename(folder))
                try:
                    os.rename(folder, renamed)
                except FileExistsError:
                    print("Error: %s already exists", renamed)
                    print("Adding timestamp to folder name")
                    renamed = os.path.join(arcpath, os.path.basename(folder) + "_" + str(os.path.getmtime(folder)))
                    os.rename(folder, renamed)
                except OSError:
                    print("Error: %s already exists", folder)
                    print("Adding timestamp to folder name")
                    renamed = os.path.join(arcpath, os.path.basename(folder) + "_" + str(os.path.getmtime(folder)))
                    os.rename(folder, renamed)

def get_config(file=path('/ML_Transcribe/config/config.json')):
    """Reads the config file and returns a dictionary"""
    print("Reading config file")
    if not os.path.exists(file):
        print("Error: %s not found", file)
        print("Warning: %s not found", file)
        print("You were probably using a symlink")
        print("Attempting to find config file")
        pwd = os.getcwd()
        if "ML_Transcribe" in pwd:
            pwd = pwd[:pwd.find("ML_Transcribe")]
            file = path(pwd + "ML_Transcribe/config/config.json")
        else:
            print("Error: Could not find config file")
            sys.exit(2)
    else:
        print("Found config file at %s", file)
    with open(file, encoding="utf-8") as conf:
        print("Loading config file")
        try:
            config = json.load(conf)
        except json.decoder.JSONDecodeError as err:
            print("Error: %s", str(err))
            sys.exit(2)
    return config, file

def create_unique_folder(data_home, fold_end):
    """Creates a unique folder for the session fold end is usually user input"""
    folder = os.path.join(data_home, fold_end, "mp3")
    seperated_folder = os.path.join(data_home, fold_end, "seperated")
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(seperated_folder):
        os.makedirs(seperated_folder)
    return folder, seperated_folder

def graph_all_wav_for_each_folder(folder, MIDI=False):
    """Graphs all .wav files in a folder Intended to be used on the seperated folder"""
    # find seperated folder
    for file in os.listdir(folder):
        if file.endswith("seperated"):
            for inner_file in os.listdir(os.path.join(folder, file)):
                if os.path.isdir(os.path.join(folder, file, inner_file)):
                    graph_all_wav(os.path.join(folder, file, inner_file), MIDI=MIDI)

def graph_all_wav(folder, MIDI=False):
    """Graphs all .wav files in a folder"""
    for file in os.listdir(folder):
        print("Graphing %s", file)
        if file.endswith(".wav"):
            graph_wav(os.path.join(folder, file))
            # MIDI is experimental and kinda trash
            if MIDI:
                wav_to_midi(os.path.join(folder, file))

def graph_wav(file):
    """
    Graphs a .wav file into three different Graphs
    1. A graph of the raw data with a frequency/amplitude graph
    2. The sound wave
    3. A Spectrogram
    """

    try:
        wav = wave.open(file, 'r')
    except FileNotFoundError:
        print("Error: %s not found", file)
        return
    frames = wav.readframes(-1)
    frame_rate = wav.getframerate()
    sound_info = np.fromstring(frames, 'int16')
    left = sound_info[::2]
    lf = np.fft.fft(left)
    wav.close()

    # Plot the raw data and the frequency/amplitude graph
    try:
        plt.figure(1)
        plt.autoscale(enable=True, axis='both', tight=None)
        a = plt.subplot(211)
        r = 2**16/2
        a.set_ylim([-r, r])
        a.set_xlabel('time [s]')
        a.set_ylabel('sample value [-]')
        x = np.arange(0, len(left), 1)
        plt.plot(x, left)
        b = plt.subplot(212)
        b.set_xscale('log')
        b.set_xlabel('frequency [Hz]')
        b.set_ylabel('|amplitude|')
        plt.plot(abs(lf))
        plt.savefig(file + '_sample-graph.png')
        plt.close()
    except Exception as err:
        print("Error: %s", str(err))
        return


    try:
        # Plot the sound wave
        fig = plt.figure(figsize=(10, 6), edgecolor='k')
        plt.title("Wav Amplitude " + str(file))
        plt.plot(sound_info)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.autoscale(tight=True)
        plt.savefig(file + "_amp.png")
        plt.close(fig)

        # Plot the spectrogram
        fig = plt.figure(figsize=(10, 6), edgecolor='k')
        plt.title("Wav Spectrogram " + str(file))
        plt.specgram(sound_info, Fs=frame_rate)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.autoscale(tight=True)
        plt.savefig(file + "_spec.png")
        plt.close(fig)
    except ValueError:
        print("Error: %s is empty", file)
    except MemoryError:
        print("Error: %s is too large", file)
    except Exception as err:
        print("Error: %s", str(err))

def cull_data_home(data_home):
    """Deletes the data home"""
    rmtree(data_home)
    make_data_home(data_home)

def make_data_home(data_home):
    """Makes the data home"""
    if not os.path.exists(data_home):
        os.makedirs(data_home)

def wav_to_midi(wav_file):
    """Converts a wav file to a midi file"""
    print("Converting %s to midi", wav_file)
    try:
        model_output, midi_data, note_activations = predict(wav_file)
    except Exception as err:
        print("Error: %s", str(err))
        return
    print("Writing midi file")
    midi_file = os.path.join(os.path.dirname(wav_file), os.path.basename(wav_file) + "_conv_midi.mid")
    midi_data.write(midi_file)

def transcribe_notes(wav_file):
    """Transcribes a wav file to notes"""
    print("Transcribing %s to notes", wav_file)


def clean_wav(seperated_folder):
    """Cleans the wav files in a folder"""
    for file in os.listdir(seperated_folder):
        if file.endswith(".wav"):
            clean_wav_file(os.path.join(seperated_folder, file))

def clean_wav_file(wav_file):
    """Cleans a wav file"""
    print("Cleaning %s", wav_file)
    try:
        wav = wave.open(wav_file, 'r')
    except FileNotFoundError:
        print("Error: %s not found", wav_file)
        return
    frames = wav.readframes(-1)
    sound_info = np.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    print("Writing cleaned wav file")
    wav_file = os.path.join(os.path.dirname(wav_file), os.path.basename(wav_file) + "_clean.wav")
    wav = wave.open(wav_file, 'w')
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(frame_rate)
    wav.writeframes(sound_info)
    wav.close()

def custom_data_transform(folder):
    """Transforms the custom data in a folder"""
    for file in os.listdir(folder):
        if file.endswith("seperated"):
            for inner_file in os.listdir(os.path.join(folder, file)):
                if os.path.isdir(os.path.join(folder, file, inner_file)):
                    custom_all_wav(os.path.join(folder, file, inner_file))

def custom_all_wav(folder):
    """Custom all .wav files in a folder"""
    for file in os.listdir(folder):
        print("Graphing %s", file)
        if file.endswith(".wav"):
            wav_to_custom(os.path.join(folder, file))

def wav_to_custom(file):
    """
    Converts a wav file to a custom format
    1. Filters data
    2. Creates data (csv) in 1/8 second intervals
        freq
        amp
        note prediction
    """
    notes = []
    print("Filtering %s", file)
    try:
        wav = wave.open(file, 'r')
        par = list(wav.getparams())
        par[3] = 0

        wav_file = os.path.join(os.path.dirname(file), os.path.basename(file) + "_filtered.wav")
        out_wav = wave.open(wav_file, 'w')
        out_wav.setparams(tuple(par))
    except FileNotFoundError:
        print("Error: %s not found", file)
        return
    framerate = wav.getframerate()
    sample_count = int(wav.getnframes()/framerate)
    for i in range(sample_count):
        print("Processing frame %s of %s", i, sample_count)
        frames = wav.readframes(framerate)
        sound_info = np.fromstring(frames, 'int16')
        sound_info, left, lf, right, rf = filter_sound_info(sound_info)
        sample_to_notes(sound_info)
        print("Writing filtered wav file")
        out_wav.writeframes(sound_info.tostring())
    wav.close()
    out_wav.close()



def get_audio_length_in_seconds(sample_rate, samples):
    """Gets the length of audio in seconds"""
    return len(samples) / sample_rate

def filter_out_low_amplitude(lf, rf):
    """Filters out low amplitude"""
    low= 21 # Remove low frequencies.
    lf[:low], rf[:low] = 0,0 # low pass filter (3)
    return lf, rf

def filter_out_high_amplitude(lf, rf):
    """Filters out high amplitude"""
    high= 9000 # Remove high frequencies.
    lf[high:], rf[high:] = 0,0 # low pass filter (3)
    return lf, rf

def filter_sound_info(sound_info):
    left = sound_info[0::2]
    right = sound_info[1::2]
    lf, rf = np.fft.fft(left), np.fft.fft(right)
    lf, rf = filter_out_low_amplitude(lf, rf)
    lf, rf = filter_out_high_amplitude(lf, rf)
    left, right = np.fft.ifft(lf), np.fft.ifft(rf)
    sound_info = np.column_stack((left, right)).ravel().astype(np.int16)
    return sound_info, left, lf, right, rf

def sample_to_notes(samples):
    """Converts a sample to notes at 1 second intervals
    Assuming samples is only one second"""
    print("Converting sample to notes")
    notes = []
    fft = np.fft.rfft(samples)
    fft = np.abs(fft)
    top_notes = find_top_notes(fft, 3)

import plotly.graph_objects as go

def plot_fft(p, xf, fs, notes, dimensions=(960,540)):
  layout = go.Layout(
      title="frequency spectrum",
      autosize=False,
      width=dimensions[0],
      height=dimensions[1],
      xaxis_title="Frequency (note)",
      yaxis_title="Magnitude",
      font={'size' : 24}
  )

  fig = go.Figure(layout=layout,
                  layout_xaxis_range=[FREQ_MIN,FREQ_MAX],
                  layout_yaxis_range=[0,1]
                  )
  
  fig.add_trace(go.Scatter(
      x = xf,
      y = p))
  
  for note in notes:
    fig.add_annotation(x=note[0]+10, y=note[2],
            text=note[1],
            font = {'size' : 48},
            showarrow=False)
  return fig

def extract_sample(audio, frame_number):
  end = frame_number * FRAME_OFFSET
  begin = int(end - FFT_WINDOW_SIZE)

  if end == 0:
    # We have no audio yet, return all zeros (very beginning)
    return np.zeros((np.abs(begin)),dtype=float)
  elif begin<0:
    # We have some audio, padd with zeros
    return np.concatenate([np.zeros((np.abs(begin)),dtype=float),audio[0:end]])
  else:
    # Usually this happens, return the next sample
    return audio[begin:end]

def find_top_notes(fft,num):
  if np.max(fft.real)<0.001:
    return []

  lst = [x for x in enumerate(fft.real)]
  lst = sorted(lst, key=lambda x: x[1],reverse=True)

  idx = 0
  found = []
  found_note = set()
  while( (idx<len(lst)) and (len(found)<num) ):
    f = xf[lst[idx][0]]
    y = lst[idx][1]
    n = freq_to_number(f)
    n0 = int(round(n))
    name = note_name(n0)

    if name not in found_note:
      found_note.add(name)
      s = [f,note_name(n0),y]
      found.append(s)
    idx += 1
    
  return found

    

        
def sample_to_note(samples):
    notes = []
    for sample in samples:
        if sample != 0:
            print("Sample: %s", sample)
            notes.append(librosa.hz_to_note(sample))
        else:
            notes.append("0")
    print("Notes: %s", notes)
    return notes


def get_freq(samples):
    """Gets the frequency of a sample"""
    freq_sum = 0
    for sample in samples:
        if sample != 0:
            freq_sum += sample
    return freq_sum / len(samples)

def get_note(freq):
    """Gets the note of a frequency"""
    return librosa.hz_to_note(freq)


def remove_non_spleeter_for_Active_folder(folder):
    """clean out old files"""
    for file in os.listdir(folder):
        if file.endswith("seperated"):
            for inner_file in os.listdir(os.path.join(folder, file)):
                if os.path.isdir(os.path.join(folder, file, inner_file)):
                    remove_non_spleeter(os.path.join(folder, file, inner_file))

def remove_non_spleeter(folder):
    """Removes non spleeter files"""
    spleeter = ("bass.wav", "drums.wav", "other.wav", "vocals.wav", "piano.wav")
    for file in os.listdir(folder):
        if file.endswith(".wav") and os.path.basename(file) not in spleeter:
            os.remove(os.path.join(folder, file))


