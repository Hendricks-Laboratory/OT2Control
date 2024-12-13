#!/bin/sh

tmux 

tmux kill-session -t run

cd OT2Control/

tmux attach -t run

git checkout main

git pull

python ot2_robot.py

tmux detach