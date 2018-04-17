source read_params.sh
pwd=$(pwd)
LOG_DIR=$(ls -td $pwd/out/* | head -1)
tensorboard --logdir=$LOG_DIR
