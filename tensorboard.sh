#jupyter nbconvert --to notebook --execute main.ipynb
pkill -f tensorboard
tensorboard --host=localhost --logdir='./train_dir/2017-12-29_13-58-45.920133/logs/' &> /dev/null &
sleep 6
open http://localhost:6006
