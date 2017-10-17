#!/bin/bash
sh clean.sh
sh run.sh
tensorboard --host=localhost --logdir='./log/'
