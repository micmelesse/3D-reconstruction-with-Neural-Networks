#jupyter nbconvert --to notebook --execute main.ipynb
pkill -f tensorboard
tensorboard --host=localhost --logdir='./logs/' &> /dev/null &
sleep 1
open http://localhost:6006
