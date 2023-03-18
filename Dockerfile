# Just latest ubuntu for now
FROM ubuntu:latest

# volume mount data folder later
ADD config /config

# Docker folder contains config and setup files
ADD docker /docker

# install dependencies
RUN apt-get update
# needed for speakers/microphone
RUN apt-get install -y alsa-base 
RUN apt-get install -y alsa-utils
RUN apt-get install -y libsndfile1-dev

# python dependencies
RUN apt-get install -y python3 python3-pip
# portaudio dependencies
RUN apt-get install -y portaudio19-dev
# ffmpeg dependencies
Run apt-get install -y ffmpeg
# install python dependencies
RUN pip3 install -r /docker/requirements.txt
# override config with a docker config
RUN cp /docker/config.json /config/config.json

# add the source files to the container
ADD src /src

#set an env variable
ENV Docker True


