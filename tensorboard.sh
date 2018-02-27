source params/logs.params
xterm -e tensorboard --logdir=$log_dir & 
sh tensorboard_vis.sh