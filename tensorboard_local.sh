source read_params.sh
pwd=$(pwd)
LOG_DIR=$(ls -td $pwd/models_local/* | head -1)
tensorboard --logdir=$LOG_DIR --reload_interval=1 --port 6006 --debugger_port 6064

