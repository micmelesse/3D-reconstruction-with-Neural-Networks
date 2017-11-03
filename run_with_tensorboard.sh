jupyter nbconvert --to notebook --execute main.ipynb
tensorboard --host=localhost --logdir='./log/'
sh clean.sh
