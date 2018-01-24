pkill -f tensorboard
tensorboard --host=localhost --logdir='./logs' &
open http://localhost:6006
