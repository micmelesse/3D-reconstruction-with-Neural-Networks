pwd=$(pwd)
LOG_DIR=$(ls -td $pwd/models_remote/* | head -1)
echo $LOG_DIR