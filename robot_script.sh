#!/bin/sh

tmux 

tmux kill-session -s run -d

tmux attach -s run

git checkout main

git pull

python ot2_robot.py

tmux detach