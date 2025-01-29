#!/bin/sh

tmux kill-session -t run

tmux new-session -s run -d 'python robot_script.py'

echo slkdf