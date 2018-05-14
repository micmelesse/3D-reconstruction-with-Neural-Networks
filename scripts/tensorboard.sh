source scripts/read_params.sh
echo $LONGEST
tensorboard --logdir=$LONGEST --reload_interval=1 

