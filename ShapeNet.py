
import os
import sys
import pandas as pd
import dataset
from filecmp import dircmp


class ShapeNet:
    def __init__(self):
        self._n = 0

    def next_batch(self, batch_count):
        self._n += batch_count

    def get_paths(self, data_dir="ShapeNetRendering", label_dir="ShapeNetVox32"):

        common_paths = []
        for dir_top, subdir_cmps in dircmp(data_dir, label_dir).subdirs.items():
            for dir_bot in subdir_cmps.common_dirs:
                common_paths.append(os.path.join(dir_top, dir_bot))

        mapping = pd.DataFrame(common_paths, columns=["common_dirs"])
        mapping['data_dirs'] = mapping.apply(
            lambda row: os.path.join(data_dir, row.common_dirs), axis=1)

        mapping['label_dirs'] = mapping.apply(
            lambda row: os.path.join(label_dir, row.common_dirs), axis=1)

        data_paths = []
        label_paths = []
        for d, l in zip(mapping.data_dirs, mapping.label_dirs):
            png_ls = dataset.construct_path_lists(d, [".png"])
            binvox_ls = dataset.construct_path_lists(l, [".binvox"])
            data_paths += png_ls
            label_paths += binvox_ls * len(png_ls)

        paths = pd.DataFrame({"data_paths": data_paths, "label_paths": label_paths})
        paths.to_csv("paths.csv")
        return paths






if __name__ == '__main__':
    shapenet = ShapeNet()
    shapenet.get_paths()
