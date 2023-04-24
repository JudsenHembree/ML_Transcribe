"""
Utility functions for the project
"""
# todo better modules it's all just in utils atm why did I do this
from pathlib import Path as path
from glob import glob
from shutil import rmtree
import sys
import os
import time
import array
import shutil
import json
import wave
import pyaudio
from tqdm import tqdm
import pretty_midi
from pretty_midi import PrettyMIDI
from matplotlib import pyplot as plt
import numpy as np
import librosa
import music21
import librosa.display
import soundfile as sf
from basic_pitch.inference import predict

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

    return config["data_home"]

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
                    shutil.move(folder, renamed)
                except FileExistsError:
                    print("Error: %s already exists", renamed)
                    print("Adding timestamp to folder name")
                    renamed = os.path.join(arcpath, os.path.basename(folder) + "_" + str(os.path.getmtime(folder)))
                    shutil.move(folder, renamed)
                except OSError:
                    print("Error: %s already exists", folder)
                    print("Adding timestamp to folder name")
                    renamed = os.path.join(arcpath, os.path.basename(folder) + "_" + str(os.path.getmtime(folder)))
                    shutil.move(folder, renamed)

def get_config(file=path('/ML_Transcribe/config/config.json')):
    """Reads the config file and returns a dictionary"""
    print("Reading config file")
    inDocker = os.environ.get("Docker", False)
    if not os.path.exists(file):
        print("Error: %s not found", file)
        print("Warning: %s not found", file)
        print("You were probably using a symlink")
        print("Attempting to find config file")
        pwd = os.getcwd()
        if "ML_Transcribe" in pwd:
            pwd = pwd[:pwd.find("ML_Transcribe")]
            file = path(pwd + "ML_Transcribe/config/config.json")
        elif inDocker:
            file = path("/config/config.json")
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

def generate_selections_for_each_folder(folder):
    """Graphs all .wav files in a folder Intended to be used on the seperated folder"""
    print("Generating selections for each wav file in %s", folder)
    # find seperated folder
    for file in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, file)):
            for inner_file in os.listdir(os.path.join(folder, file)):
                print(inner_file)
                if inner_file.endswith(".wav"):
                    generate_selection(os.path.join(folder, file, inner_file))

def generate_selection(file):
    """Generates a selection for a wav file"""
    print("Generating selection for %s", file)
    head, tail = os.path.split(file)
    head = os.path.join(head, "selections")
    if not os.path.exists(head):
        os.makedirs(head)
    head = os.path.join(head, tail)
    print("Saving selection to %s", head)
    hanning(file, head)


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

def cull_data_home(data_home, remake=True):
    """Deletes the data home"""
    rmtree(data_home)
    if remake:
        make_data_home(data_home)

def make_data_home(data_home):
    """Makes the data home"""
    if not os.path.exists(data_home):
        os.makedirs(data_home)

# kinda trash not really pursing this atm
def wav_to_midi(wav_file, dest = ""):
    """Converts a wav file to a midi file"""
    print("Converting %s to midi", wav_file)
    try:
        model_output, midi_data, note_activations = predict(wav_file)
    except Exception as err:
        print("Error: %s", str(err))
        return
    print("Writing midi file")
    if dest == "":
        midi_file = os.path.join(os.path.dirname(wav_file), os.path.basename(wav_file) + "_conv_midi.mid")
    else:
        if not os.path.isdir(dest + "/midi"):
            os.makedirs(dest + "/midi/piano")
            os.makedirs(dest + "/midi/vocals")
            os.makedirs(dest + "/midi/drums")
            os.makedirs(dest + "/midi/bass")
            os.makedirs(dest + "/midi/other")
        elif not os.path.isdir(dest + "/midi/piano") or not os.path.isdir(dest + "/midi/vocals") or not os.path.isdir(dest + "/midi/drums") or not os.path.isdir(dest + "/midi/bass") or not os.path.isdir(dest + "/midi/other"):
            os.makedirs(dest + "/midi/piano")
            os.makedirs(dest + "/midi/vocals")
            os.makedirs(dest + "/midi/drums")
            os.makedirs(dest + "/midi/bass")
            os.makedirs(dest + "/midi/other")
        instrument = "bass"
        if "piano" in wav_file:
            instrument = "piano"
        elif "vocals" in wav_file:
            instrument = "vocals"
        elif "drums" in wav_file:
            instrument = "drums"
        elif "other" in wav_file:
            instrument = "other"
        midi_file = os.path.join(dest + "/midi/" + instrument + "/" + os.path.basename(wav_file) + "_conv_midi.mid")
    midi_data.write(midi_file)

def clean_midi(midi_file, piano_roll = False):
    """Cleans a midi file, and creates an image output"""
    print("Cleaning %s", midi_file)
    try:
        midi = PrettyMIDI(midi_file)
    except FileNotFoundError:
        print("Error: %s not found", midi_file)
        return
    for instrument in midi.instruments:
        print(instrument)
    if piano_roll:
        fig = plt.figure(figsize=(12, 4))
        plot_piano_roll(midi, 24, 84)
        plt.savefig(midi_file + "_piano_roll.png")
        plt.close(fig)

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

def clean_wav(seperated_folder):
    """Cleans the wav files in a folder"""
    for file in os.listdir(seperated_folder):
        if file.endswith(".wav"):
            clean_wav_file(os.path.join(seperated_folder, file))

def clean_wav_file(wav_file, replace = False):
    """Cleans a wav file and removes low amplitude"""
    print("Cleaning %s", wav_file)
    try:
        wav = wave.open(wav_file, 'r')
    except FileNotFoundError:
        print("Error: %s not found", wav_file)
        return
    framerate = wav.getframerate()
    frames = wav.readframes(framerate*10)
    sound_info = np.fromstring(frames, 'int16')
    sound_info, left, lf, right, rf = filter_sound_info(sound_info)
    wav.close()
    if replace:
        wav = wave.open(wav_file, 'w')
        wav.setparams((2, 2, framerate, 10, 'NONE', 'not compressed'))
        wav.writeframes(sound_info)
        wav.close()
    else:
        wav = wave.open(wav_file + "_cleaned.wav", 'w')
        wav.setparams((2, 2, framerate, 10, 'NONE', 'not compressed'))
        wav.writeframes(sound_info)
        wav.close()

# def custom_data_transform(folder):
#     """Transforms the custom data in a folder"""
#     for file in os.listdir(folder):
#         if file.endswith("seperated"):
#             for inner_file in os.listdir(os.path.join(folder, file)):
#                 if os.path.isdir(os.path.join(folder, file, inner_file)):
#                     custom_all_wav(os.path.join(folder, file, inner_file))
# 
# def custom_all_wav(folder):
#     """Custom all .wav files in a folder"""
#     for file in os.listdir(folder):
#         print("Graphing %s", file)
#         if file.endswith(".wav"):
#             wav_to_custom(os.path.join(folder, file))
# 
# def wav_to_custom(file):
#     """
#     Converts a wav file to a custom format
#     1. Filters data
#     2. Creates data (csv) in 1/8 second intervals
#         freq
#         amp
#         note prediction
#     """
#     notes = []
#     print("Filtering %s", file)
#     try:
#         wav = wave.open(file, 'r')
#         par = list(wav.getparams())
#         par[3] = 0
# 
#         wav_file = os.path.join(os.path.dirname(file), os.path.basename(file) + "_filtered.wav")
#         out_wav = wave.open(wav_file, 'w')
#         out_wav.setparams(tuple(par))
#     except FileNotFoundError:
#         print("Error: %s not found", file)
#         return
#     framerate = wav.getframerate()
#     sample_count = int(wav.getnframes()/framerate)
#     for i in range(sample_count):
#         print("Processing frame %s of %s", i, sample_count)
#         frames = wav.readframes(framerate)
#         sound_info = np.fromstring(frames, 'int16')
#         sound_info, left, lf, right, rf = filter_sound_info(sound_info)
#         # TODO Get notes from sound_info
#         sample_to_notes(sound_info)
#         print("Writing filtered wav file")
#         out_wav.writeframes(sound_info.tostring())
#     wav.close()
#     out_wav.close()

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
    high= 20000 # Remove high frequencies.
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

# todo find notes
def find_top_notes(fft, top):
    """Finds the top notes in a fft"""
    top_notes = []
    for i in range(top):
        top_notes.append(np.argmax(fft))
        fft[top_notes[i]] = 0
    print("Top notes: %s", top_notes)
    return top_notes

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

def hanning(wav_file, dest):
    """Hanning window"""
    with wave.open(wav_file, 'rb') as f:
        nchannels, sampwidth, framerate, nframes, comptype, compname = f.getparams()
        assert sampwidth == 2  # 16-bit WAV files only
        data = f.readframes(nframes)

    if nchannels == 2:
        # just use the left
        data = data[::2]
    # Convert the data to a numpy array
    y = np.frombuffer(data, dtype=np.int16)

    # Set the length of the Hanning window to 10 seconds
    window_length = int(framerate * 10)
    print(f"Window length: {window_length}")

    # Divide the audio signal into 10-second windows
    num_windows = len(y) // window_length
    print(f"Number of windows: {num_windows}")
    print(f"Number of samples per window: {window_length}")
    windows = np.array_split(y[:num_windows * window_length], num_windows)

    # Compute the FFT and average amplitude for each window
    avg_amplitudes = []
    for window in windows:
        window_hann = window * np.hanning(window_length)
        window_fft = np.abs(np.fft.fft(window_hann))
        window_amplitude = np.mean(window_fft[:window_length//2])
        avg_amplitudes.append(window_amplitude)

    # Find the index of the window with the highest average amplitude
    max_index = np.argmax(avg_amplitudes)

    # Convert the index to a start time
    start_time = max_index * window_length / framerate

    # Print the start and end times of the window with the highest average amplitude
    print(f"Window with highest average amplitude: {start_time:.2f}s - {(start_time+10):.2f}s")

    # load the converted wav
    wf = wave.open(wav_file, 'rb')
    # open the stream
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # capture played audio to record the selected 10 seconds
    selected_frames = []
    selected_fr = wf.getframerate()
    selected_format = p.get_format_from_width(wf.getsampwidth())
    selected_channels = wf.getnchannels()
    chunk = 1024
    duration = 10 # seconds
    data = array.array('h')

    # skip to start
    wf.setpos(int(start_time * wf.getframerate()))
    for _ in tqdm(range(0, int(wf.getframerate() / chunk * duration)), colour='green'):
        data = wf.readframes(chunk)
        selected_frames.append(data)

    # close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # save the selected audio
    wf = wave.open(dest, 'wb')
    wf.setnchannels(selected_channels)
    wf.setsampwidth(p.get_sample_size(selected_format))
    wf.setframerate(selected_fr)
    wf.writeframes(b''.join(selected_frames))
    wf.close()

# def play_file_starting_at(file, start, duration, split):
#     """plays a wav starting at start for duration"""
#     print("Playing file")
# 
#     if split == "other":
#         print("Skipping other")
#         return
# 
#     # try to find the original audio file
#     # get head and base of file
#     head, base = os.path.split(file)
#     # swap head to mp3 folder instead of seperated
#     head = head.replace("seperated", "wav")
#     globbed_song = glob(head + ".wav")
#     if len(globbed_song) > 0:
#         file = globbed_song[0]
#         print(f"Found original file: {file}")
#     else:
#         print(f"Could not find original file: {file}")
#         return
# 
#     # for recording
#     chunk = 1024
#     sample_format = pyaudio.paInt16
#     channels = 2
#     fs = 44100
#     seconds = 10
#     filename = "recorded_" + split + ".wav"
# 
#     p_record = pyaudio.PyAudio()
# 
#     stream_record = p_record.open(format=sample_format,
#                     channels=channels,
#                     rate=fs,
#                     frames_per_buffer=chunk,
#                     input=True)
# 
#     frames = []
# 
#     # load the converted wav
#     wf = wave.open(file, 'rb')
#     # open the stream
#     p = pyaudio.PyAudio()
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=wf.getframerate(),
#                     output=True)
# 
#     # capture played audio to record the selected 10 seconds
#     selected_frames = []
#     selected_fr = wf.getframerate()
#     selected_format = p.get_format_from_width(wf.getsampwidth())
#     selected_channels = wf.getnchannels()
# 
# 
#     print("Mimic the following audio focusing on the " + str(split) + " part.")
#     while frames == []:
#         # skip to start
#         wf.setpos(int(start * wf.getframerate()))
#         # ask user to press enter when ready
#         option = input("Choose an option:\n\t1. Play without recording for practice\n\t2. Play with recording.\n")
#         if option == "1":
#             # play for duration
#             for _ in tqdm(range(0, int(wf.getframerate() / chunk * duration)), colour='green'):
#                 data = wf.readframes(chunk)
#                 stream.write(data)
#         elif option == "2":
#             # start Recording
#             for _ in tqdm(range(0, int(wf.getframerate() / chunk * duration)), colour='green'):
#                 data = wf.readframes(chunk)
#                 stream.write(data)
#                 selected_frames.append(data)
# 
#                 record_data = stream_record.read(chunk)
#                 frames.append(record_data)
#         else:
#             print("Invalid option")
# 
#     # close the stream
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
# 
#     # stop recording
#     stream_record.stop_stream()
#     stream_record.close()
#     p_record.terminate()
# 
#     # save the Recording
#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(channels)
#     wf.setsampwidth(p.get_sample_size(sample_format))
#     wf.setframerate(fs)
#     wf.writeframes(b''.join(frames))
#     wf.close()
# 
#     # save the selected audio
#     wf = wave.open("selected.wav", 'wb')
#     wf.setnchannels(selected_channels)
#     wf.setsampwidth(p.get_sample_size(selected_format))
#     wf.setframerate(selected_fr)
#     wf.writeframes(b''.join(selected_frames))
#     wf.close()
# """

def is_too_quiet(file, min_amplitude):
    """makes sure there is audio"""
    # get the mean Amplitude
    wav, _ = librosa.load(file, sr=44100)
    mean_amplitude = np.mean(np.abs(wav))
    print(f"Mean amplitude: {mean_amplitude}")
    if mean_amplitude < min_amplitude:
        return True
    return False

def record(folder):
    """for every song record hum along with selected 10 seconds"""
    # if prior recording exist nuke them
    if os.path.exists(folder + "/recordings"):
        shutil.rmtree(folder + "/recordings")
    pth = path(folder + "/seperated/*/selections/*.wav")
    globbed = glob(str(pth))
    if len(globbed) == 0:
        print("couldn't find selections.")
        return
    piano_count = 0
    drum_count = 0
    vocal_count = 0
    bass_count = 0
    for file in globbed:
        # make the recording folder if needed
        outwav = file.replace("selections", "recordings")
        head, base = os.path.split(outwav)
        if not os.path.exists(head):
            os.makedirs(head)
        else:
            # else nuke it
            shutil.rmtree(head)
            os.makedirs(head)
        # if other or piano just skip
        if base == "other.wav" or base == "piano.wav":
            continue

    for file in globbed:
        # make the recording folder if needed
        outwav = file.replace("selections", "recordings")
        head, base = os.path.split(outwav)
        # if other just skip
        if base == "other.wav" or base == "piano.wav":
            continue
        if "piano" in file:
            piano_count += 1
            if piano_count > 10:
                print("Skipping piano")
                continue
        elif "drums" in file:
            drum_count += 1
            if drum_count > 10:
                print("Skipping drums")
                continue
        elif "vocals" in file:
            vocal_count += 1
            if vocal_count > 10:
                print("Skipping vocals")
                continue
        elif "bass" in file:
            bass_count += 1
            if bass_count > 10:
                print("Skipping bass")
                continue
        wf = wave.open(file, 'rb')
        # if the audio is too quiet then skipping
        if is_too_quiet(file, 0.001):
            print(f"Skipping {file} because it is too quiet")
            continue
        # open the stream
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # for recording
        record_frames = []
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = wf.getnchannels()
        fs = 44100
        seconds = 10
        outwf = wave.open(outwav, 'wb')
        outwf.setnchannels(channels)
        outwf.setsampwidth(p.get_sample_size(sample_format))
        outwf.setframerate(fs)

        p_record = pyaudio.PyAudio()

        print("completed " + str(bass_count) + " bass recordings")
        print("completed " + str(vocal_count) + " vocal recordings")
        print("completed " + str(drum_count) + " drum recordings")
        selection = "" 
        while selection != "2":
            selection = input("Choose an option:\n\t1. Play without recording for practice\n\t2. Play with recording.\n\t3. skip this recording.\n\tOption: ")
            if selection == "2":
                print("Mimic the following audio focusing on the " + str(base.split(".")[0]) + " part.")
                input("Press enter when ready")
                stream_record = p_record.open(format=sample_format,
                                channels=2,
                                rate=fs,
                                frames_per_buffer=chunk,
                                input=True)
                #sleep for a second to let the audio open
                time.sleep(0.1)
                #TODO glitchy not sure why
                #TODO poppy in Docker idk why
                record_data = array.array('h')
                for _ in tqdm(range(0, int(wf.getframerate() / chunk * seconds)), colour='green'):
                    data = wf.readframes(chunk)
                    stream.write(data)
                    # record
                    # need exception_on_overflow for docker (THERE IS SOME BUG HERE)

                    record_data = stream_record.read(chunk, exception_on_overflow = False)
                    record_frames.append(record_data)
                stream_record.stop_stream()
                stream_record.close()
            elif selection == "1":
                print("Playing the following audio focusing on the " + str(base.split(".")[0]) + " part.")
                input("Press enter when ready")
                for _ in tqdm(range(0, int(wf.getframerate() / chunk * seconds)), colour='green'):
                    data = wf.readframes(chunk)
                    stream.write(data)
            elif selection == "3":
                print("Skipping this track")
                break
            else:
                print("Invalid option")
            wf.rewind()

        # save the Recording
        outwf.writeframes(b''.join(record_frames))

        # close the streams
        stream.stop_stream()
        stream.close()
        p.terminate()
        p_record.terminate()
        outwf.close()

        if selection == "3":
            os.remove(outwav)
            if "piano" in file:
                piano_count -= 1
            elif "drums" in file:
                drum_count -= 1
            elif "vocals" in file:
                vocal_count -= 1
            elif "bass" in file:
                bass_count -= 1

def convert_to_wav(file):
    """Converts a file to wav"""
    print("Converting to wav")
    # get head and base of file
    head, base = os.path.split(file)
    # swap head to wav folder instead of mp3
    head = head.replace("mp3", "wav")
    # make the new file name
    new_file = os.path.join(head, base.replace(".mp3", ".wav"))
    # make the new folder
    os.makedirs(head, exist_ok=True)
    # convert
    sound, sr = librosa.load(file, sr=44100)
    sf.write(new_file, sound, sr)
    return new_file


def convert_all_mp3_to_wav(folder):
    """Converts all mp3 files in a folder to wav"""
    print("Converting all mp3 to wav")
    print(f"Folder: {folder}")
    pth = path(folder + "/*.mp3")
    globbed = glob(str(pth))
    for file in globbed:
        print(f"Converting: {file}")
        convert_to_wav(file)

    
def collect_recordings_place_in_folder(folder):
    """Collects all recorded audio and places them in a folder for each stem"""
    print("Gathering resources")
    # get all the recordings
    stems = ("bass", "drums", "other", "vocals", "piano")
    for stem in stems:
        pth = path(folder + "/seperated/*/recordings/" + stem + ".wav")
        recordings = glob(str(pth))
        print(str(stem) + ": " + str(recordings))
        # make the folders
        try:
            dir_pth = path(folder + "/recordings/" + stem)
            os.makedirs(dir_pth, exist_ok=True)
        except PermissionError:
            print("Permission denied")
            print("Claiming the data")
            os.system("sudo chown -R $USER:$USER " + folder)
            dir_pth = path(folder + "/recordings/" + stem)
            os.makedirs(dir_pth, exist_ok=True)

        # copy the recordings
        # use a count to differentiate
        count = 0
        for recording in recordings:
            print("Copying " + recording)
            cpy_pth = path(folder + "/recordings/" + stem + "/" + str(count) + ".wav")
            shutil.copy(recording, cpy_pth)
            count += 1

def generate_meta_data(folder):
    """create the csv for the CNN"""
    print("Generating metadata")
    csv_pth = path(folder + "/metadata.csv")
    csv = open(csv_pth, "w")
    csv.write("filename,class_id,stem\n")
    stems = (("bass", 0), ("drums", 1), ("vocals", 2), ("piano", 3), ("other", 4), )
    for stem in stems:
        try:
            recordings = glob(str(path(folder + "/recordings/" + stem[0] + "/*.wav")))
        except FileNotFoundError:
            print("no recordings")
            return
        if len(recordings) == 0:
            print("No recordings found for " + stem[0] + " skipping")
            continue
        for recording in recordings:
            csv.write(recording + "," + str(stem[1]) + "," + stem[0] + "\n")
    csv.close()
    return str(path(folder + "/metadata.csv"))

def generate_CNN_inputs(folder):
    """for each stem generate spectrograms for the cnn"""
    print("Making this an image problem.")
    if "recordings" not in os.listdir(folder):
        print("you need to record first")
        # make this throw and kill program later
        return
    if "spectrograms" not in os.listdir(folder):
        os.makedirs(str(path(folder + "/spectrograms", exist_ok=True)))
    for stem in os.listdir(str(path(folder + "/recordings"))):
        if stem not in os.listdir(str(path(folder + "/spectrograms"))):
            os.makedirs(str(path(folder + "/spectrograms/" + stem)), exist_ok=True)
        # glob the recordings then loop
        globbed = glob(str(path(folder + "/recordings/" + stem + "/*.wav")))
        for recording in globbed:
            # make the spectrogram
            head, base = os.path.split(recording)
            # make the new file name
            new_file = os.path.join(head, base.replace(".wav", ".png"))
            new_file = new_file.replace("recordings", "spectrograms")
            # create the spectrogram
            sample, sr = librosa.load(recording, sr=44100)
            mel = librosa.feature.melspectrogram(y=sample, sr=sr)
            fig, ax = plt.subplots()
            mel_dB = librosa.power_to_db(mel, ref=np.max)
            img = librosa.display.specshow(mel_dB, x_axis='time', y_axis='mel', sr=sr, 
                                           fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            plt.savefig(new_file)

            plt.close()


def legacy(data_home):
    """Moves the legacy split audio into data"""
    print("Moving legacy data")
    legacy_data = data_home.replace("data", "legacy_data")
    # remove current data folder and remake it empty
    cull_data_home(data_home, remake=False)
    # copy legacy data to data
    shutil.copytree(legacy_data, data_home)

def record_for_test(outwav):
    """ record 10 seconds of audio for testing"""
    if not os.path.exists(outwav):
        os.makedirs(outwav)
    p_record = pyaudio.PyAudio()
    # for recording
    record_frames = []
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100
    seconds = 10

    selection = input("Press enter to begin recording")
    stream_record = p_record.open(format=sample_format,
                    channels=2,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    #sleep for a second to let the audio open
    time.sleep(0.1)
    #TODO glitchy not sure why
    #TODO poppy in Docker idk why
    record_data = array.array('h')
    for _ in tqdm(range(0, int(44100 / chunk * seconds)), colour='green'):
        # record
        # need exception_on_overflow for docker (THERE IS SOME BUG HERE)
        record_data = stream_record.read(chunk, exception_on_overflow = False)
        record_frames.append(record_data)
    stream_record.stop_stream()
    stream_record.close()

    outwf = wave.open(os.path.join(outwav, "recording.wav"), 'wb')
    outwf.setnchannels(channels)
    outwf.setsampwidth(p_record.get_sample_size(sample_format))
    outwf.setframerate(fs)
    outwf.writeframes(b''.join(record_frames))
    outwf.close()

