#!/usr/bin/env bash

python -m visdom.server -port=12306 &

python -u vis.py &