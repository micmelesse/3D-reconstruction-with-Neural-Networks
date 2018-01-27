git pull
screen -ls
python lib/prep_to_train.py
sh to_script.sh demo
python demo.py
# sh send_email.sh