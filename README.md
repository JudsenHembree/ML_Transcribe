# ML_Transcribe
## Setup
I'm assuming you run python3
You need to install
- spleeter
```
pip install spleeter
```
- spotdl
```
pip install spotdl
```

The code uses a symlink to this repo.
In terminal go to one level above the repo and link it
On linux in bash
```
sn -s $(pwd)/ML_Transcribe /ML_Transcribe
```
Or you can just manually do the full path without the pwd thing.

## repo structure
For each folder
- config
    - config.json for program behavior. Where to route .mp3s etc.
    - put stuff here as needed
- data
    - download mp3s to a folder specified by user input here.
- src
    - python app
    - utils module

## How to run
There is a makefile if you want
- clear out the data folder
```
make clean
```
- do a run
```
make run
```

You can obviously not use the makefile
```
python src/app.py
```
![demo](./gifs/demo.gif)
