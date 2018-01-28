if [ $EUID != 0 ]; then
    sudo "$0" "$@"
    exit $?
fi

sh prep_to_train.sh
sh download_dataset.sh
python -c "from lib import dataset
if __name__ == '__main__':
    dataset.main()"