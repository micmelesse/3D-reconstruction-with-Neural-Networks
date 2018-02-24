source config/logs.params
xterm -e tensorboard --logdir=$d2 & 
sh tensorboard_vis.sh