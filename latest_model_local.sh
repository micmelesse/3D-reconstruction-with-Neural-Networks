pwd=$(pwd)
LATEST_LOCAL=$(ls -td $pwd/models_local/* | head -1)
echo $LOG_DIR