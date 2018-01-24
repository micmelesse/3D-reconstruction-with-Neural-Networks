# source train.params
git pull
screen -ls
sh convert_to_script.sh demo
python demo.py #$learning_rate $batch_size $num_of_epochs
sh send_email.sh