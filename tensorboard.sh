source read_params.sh
echo $LOG_DIR
tensorboard --logdir=$LOG_DIR --reload_interval=1 

