sh scripts/setup_dir.sh
sh scripts/download_dataset.sh
sh scripts/clean_csv.sh
python -c "from lib import dataset
if __name__ == '__main__':
    dataset.create_path_csv('data/ShapeNetRendering','data/ShapeNetVox32')
    "