git pull
screen -ls
sh convert_to_script.sh demo
python main.py #$learning_rate $batch_size $num_of_epochs
sh send_email.sh