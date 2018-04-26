sh setup_dir.sh
sh download_dataset.sh
sh clean_preprocessed_data.sh
python -c "from lib import dataset
if __name__ == '__main__':
    dataset.preprocess_dataset()"