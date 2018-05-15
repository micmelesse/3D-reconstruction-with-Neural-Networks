source scripts/read_params.sh
echo $1 $2
tensorboard --logdir=$1 --reload_interval=1 --port $2
