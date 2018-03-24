sh prep_dir.sh
sh download_dataset.sh
sh clean_csv.sh
python -c "from lib import path
if __name__ == '__main__':
    path.write_path_csv('data/ShapeNetRendering','data/ShapeNetVox32')
    "