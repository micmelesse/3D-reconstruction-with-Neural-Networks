#jupyter nbconvert --to notebook --execute main.ipynb
tensorboard --host=localhost --logdir='./logs/' &> /dev/null &
