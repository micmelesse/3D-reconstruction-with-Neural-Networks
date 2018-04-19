source read_params.sh
pwd=$(pwd)
LOG_DIR=$(ls -td $pwd/aws/* | head -1)
open http://localhost:6006/
tensorboard --logdir=$LOG_DIR
