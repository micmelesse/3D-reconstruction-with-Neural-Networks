#!/bin/bash
python main.py
tensorboard --host=localhost --logdir='./log/'
