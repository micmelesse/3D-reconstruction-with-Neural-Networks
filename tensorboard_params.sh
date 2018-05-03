sh read_param.sh
BEST=$(jq -r '.SESSIONS.BEST' params.json)
LONGEST=$(jq -r '.SESSIONS.LONGEST' params.json)
LSTM=$(jq -r '.SESSIONS.LSTM' params.json)
CUR_DIR=$(jq -r '.SESSIONS.CUR_DIR' params.json)

tensorboard --logdir=$CUR_DIR --reload_interval=1 

