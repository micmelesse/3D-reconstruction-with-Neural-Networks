pwd=$(pwd)
LOG_DIR=$(ls -td $pwd/models_local/* | head -1)
echo $LOG_DIR