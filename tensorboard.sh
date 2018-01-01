#jupyter nbconvert --to notebook --execute main.ipynb
pkill -f tensorboard
tensorboard --host=localhost --logdir='./logs' &> /dev/null &
open http://localhost:6006
