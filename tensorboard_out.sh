source read_params.sh
pwd=$(pwd)
LOG_DIR=$(ls -td $pwd/out/* | head -1)
open http://localhost:6006/
tensorboard --logdir=$LOG_DIR

