#!/bin/bash
sh clear.sh
jupyter nbconvert --to notebook --execute demo.ipynb
sh tensorboard.sh
