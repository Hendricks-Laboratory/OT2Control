#!/bin/sh

tmux kill-session -t run

tmux new-session -s run -d

echo slkdf