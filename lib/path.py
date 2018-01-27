import os
import sys
import pandas as pd
from filecmp import dircmp


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
    for i, d, l in zip(common_paths, mapping.data_dirs, mapping.label_dirs):
        data_row = []
        data_row += construct_path_lists(d, [".png"])
        data_row += construct_path_lists(l, [".binvox"])
        data_row += [i]
        table.append(data_row)

    paths = pd.DataFrame(table)
    paths.to_csv("out/paths.csv")
    return paths


def construct_path_lists(data_dir, file_types):
    print("[construct_path_lists] parsing dir {} for {} ...".format(
        data_dir, file_types))
    
    if isinstance(file_types, str):
        file_types = [file_types]
    paths = [[] for _ in range(len(file_types))]
    
    for root, _, files in os.walk(data_dir):
        for f_name in files:
            for i, f_type in enumerate(file_types):
                if f_name.endswith(f_type):
                    (paths[i]).append(root + '/' + f_name)

    if len(file_types) == 1:
        return paths[0]

    return tuple(paths)
