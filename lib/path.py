import os
import sys
import pandas as pd
from filecmp import dircmp


def construct_path_lists(data_dir, file_filter):
    if isinstance(file_filter, str):
        file_filter = [file_filter]
    paths = [[] for _ in range(len(file_filter))]

    for root, _, files in os.walk(data_dir):
        for f_name in files:
            for i, f_substr in enumerate(file_filter):
                if f_substr in f_name:
                    (paths[i]).append(root + '/' + f_name)

    if len(file_filter) == 1:
        return paths[0]

    return tuple(paths)


def write_path_csv(data_dir, label_dir):
    print("creating path csv for {} and {}".format(data_dir, label_dir))

    common_paths = []
    for dir_top, subdir_cmps in dircmp(data_dir, label_dir).subdirs.items():
        for dir_bot in subdir_cmps.common_dirs:
            common_paths.append(os.path.join(dir_top, dir_bot))

    mapping = pd.DataFrame(common_paths, columns=["common_dirs"])
    mapping['data_dirs'] = mapping.apply(
        lambda data_row: os.path.join(data_dir, data_row.common_dirs), axis=1)

    mapping['label_dirs'] = mapping.apply(
        lambda data_row: os.path.join(label_dir, data_row.common_dirs), axis=1)

    table = []
    for n, d, l in zip(common_paths, mapping.data_dirs, mapping.label_dirs):
        data_row = [os.path.dirname(n)+"_"+os.path.basename(n)]
        data_row += construct_path_lists(d, [".png"])
        data_row += construct_path_lists(l, [".binvox"])
        table.append(data_row)

    paths = pd.DataFrame(table)
    paths.to_csv("out/paths.csv")
    return paths
