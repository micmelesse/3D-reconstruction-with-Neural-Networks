source read_params.sh
pwd=$(pwd)
LOG_DIR=$(ls -td $pwd/model_remote/* | head -1)
tensorboard --logdir=$LOG_DIR --reload_interval=1
