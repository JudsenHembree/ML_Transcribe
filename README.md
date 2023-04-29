# ML_Transcribe
## FOR THE GRADER/HOW TO REPRODUCE
First please clone this repo so you have access to things like the makefile. 
I would not try to record audio if I were you it's laborious and liable to fail if not running ubuntu. 
### Recommended Docker
- I have a docker image for the project I would reccomend using that to recreate results since it will give 
you an exact copy of my project where I know that it works.
- docker pull feastertas/ml_transcribe
- make run should launch the image. (DISCLAIMER: I USE UBUNTU YOU MAY NEED TO REMOVE THE '--device /dev/snd' PART IT MAY WORK DIFFERENTLY ON OTHER OPERATING SYSTEMS)
- once inside the interactive image
    1. python3 src/app.py --help            <--- for help
    2. python3 src/app.py --train            <--- for training a single model
    3. python3 src/app.py --train_variable_layers            <--- for training a models with a variety of layer setups

### Run local with dataset from git LFS
If you don't want to use docker (which I would reccomend once again) the github has lfs setup as well.
1. get lfs if you don't have it. [LFS](https://git-lfs.com/)
2. Once you have downloaded git lfs run (git lfs install) <-- sets up account
3. get the dataset and pull it by (git lfs fetch --all)

You *should* now have all the dataset you can then use the following to run.
1. python3 src/app.py --help            <--- for help
2. python3 src/app.py --train            <--- for training a single model
3. python3 src/app.py --train_variable_layers            <--- for training a models with a variety of layer setups

## Current state of project
1. pull audio from a spotify link
2. for said audio seperate it into 5 stem model
3. clean it up a bit
4. pull out highest average amplitude section of the split audio. (what part of the song has a lot of piano...)
5. Play the selected audio and record audio as the user hums/sings/something along
6. Take selected audio and generate mel spectrograms.
7. Feed those into a cnn for image classification (aka sound classification)
8. Use spotifies Basic_Pitch module to generate midi based off of the recording. 
### TODO
1. Note Transcription (JUD) (COMPLETED, BUT NAIVE)
2. Many parameter runs (programatic) (JONAH)
    - Graph results 
3. Re-test on a large Dataset (JUD) :heavy_check_mark:
4. Windows tests (JUD/JONAH) :heavy_check_mark:
5. Python modularize (JONAH)
    - IF NOT DO SOME KIND OF SETUP.SH TO MAKE IT EASY FOR GRADER TO RUN.
6. Docker (JUD) (:heavy_check_mark:, but can only record on linux distros)
## Setup
### Local (on your own machine)
- I'm assuming you run python3
- If you'd like to run locally the module requirements are located at "docker/requirements.txt"
- Everything was tested on Ubuntu

```
pip install -r docker/requirements.txt
```

#### Running on windows
On windows symlinks are a pain. 
I have the code try to find your full path and set up the config based off of that. 
*Should* work out of the box. 

### Docker
- I've set up a docker container if you'd like to use that
- There is a makefile for you. Simply 
``` 
make run
```
then
```
python3 src/app.py --help
```
#### Problems
1. Using Mics/speakers in docker is weird. (It sort of works on Ubuntu, but I wouldn't record any audio here myself.)

## repo structure
For each folder
- config
    - config.json for program behavior. Where to route .mp3s etc.
    - put stuff here as needed
- data
    - some folder name (active folder)
        - where the seperated audio lands. 
        - mp3s
            - download mp3s to a folder specified by user input here.
    - archive
        - prior runs (differentiated by timestamp if same folder name)
- src
    - python app
    - utils module

## Demo gifs
![demo](./gifs/demo.gif)

## Disclamers
Sometimes spleeter fails and not every file survives running. There is no guarantee that a playlist of 30 songs results in a dataset of 30 songs. 
