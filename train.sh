screen -ls
sh convert_to_script.sh train
python train.py
sh send_email.sh