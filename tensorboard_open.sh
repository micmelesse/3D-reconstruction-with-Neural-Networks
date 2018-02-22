source config/dir.params
xterm -e tensorboard --logdir=$LOG_DIR & 
sh tensorboard_vis.sh