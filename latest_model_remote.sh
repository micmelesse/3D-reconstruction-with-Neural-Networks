pwd=$(pwd)
LATEST_REMOTE=$(ls -td $pwd/models_remote/* | head -1)
echo $LOG_DIR