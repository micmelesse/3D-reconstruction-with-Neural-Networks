source read_params.sh
pwd=$(pwd)
LOG_DIR=$(ls -td $pwd/aws/* | head -1)
tensorboard --logdir=$LOG_DIR &
open http://localhost:6006/
