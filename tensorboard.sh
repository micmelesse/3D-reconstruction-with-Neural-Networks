source params/logs.params
tensorboard --logdir=$log_dir
sh tensorboard_vis.sh