FROM ubuntu:18.04

RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python3-pip zlib1g-dev cmake libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1 python-opengl
ENV CODE_DIR /root/code
ENV VENV /root/venv


RUN apt-get install -y python-pip
RUN \
    pip install virtualenv && \
    virtualenv $VENV --python=python3 && \
    . $VENV/bin/activate && \
    mkdir $CODE_DIR && \
    cd $CODE_DIR && \
    pip install --upgrade pip && \
    pip install codacy-coverage && \
    pip install scipy && \
    pip install tqdm && \
    pip install joblib && \
    pip install zmq && \
    pip install dill && \
    pip install progressbar2 && \
    pip install mpi4py && \
    pip install cloudpickle && \
    pip install tensorflow==1.5.0 && \
    pip install click && \
    pip install opencv-python && \
    pip install numpy && \
    pip install pandas && \
    pip install pytest==3.5.1 && \
    pip install pytest-cov && \
    pip install pytest-env && \
    pip install pytest-xdist && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install glob2 && \
    pip install gym[atari,classic_control]>=0.10.9 && \
	pip install gym && \
	pip install stable-baselines && \
	pip install opencv-python && \
	pip install gym-retro && \
	pip install pynput

ENV PATH=$VENV/bin:$PATH

WORKDIR /home/root/


# HOW TO
# docker build -t rl_sonic .
# docker run -it --rm -v <PATH on drive>:/home/root/digitalfestival -e DISPLAY=<hostname>:0.0 -p 6006:6006 --ipc=host --name test rl_sonic /bin/bash
# cd digitalfestival
# python -m retro.import roms
# python . --train --render --envs 1 --state GreenHillZone.Act1
# python . --retrain --model green_hill_2.pkl --render --envs 1 --state GreenHillZone.Act1
# python . --eval --render --model green_hill_2.pkl --envs 1 --state GreenHillZone.Act2