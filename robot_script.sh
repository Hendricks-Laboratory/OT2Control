#!/bin/sh

tmux 

tmux kill-session -t run -d

tmux attach -t run

git checkout main

git pull

python ot2_robot.py

tmux detach