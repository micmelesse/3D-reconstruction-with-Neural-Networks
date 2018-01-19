#
screen -ls
sh convert_to_script.sh train
python train.py #$learning_rate $batch_size $num_of_epochs
sh send_email.sh