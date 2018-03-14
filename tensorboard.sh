source params/logs.params
tensorboard --logdir=$log_dir &
sh localhost_tensorboard.sh