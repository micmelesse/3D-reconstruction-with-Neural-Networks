source latest_model_remote.sh
tensorboard --logdir=$LOG_DIR --reload_interval=1  --port 6006 --debugger_port 6064
