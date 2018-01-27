import os


def check(train_dir):
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)


def main():
    TRAIN_DIRS = ["out", "config", "data", "aws"]
    for d in TRAIN_DIRS:
        check(d)
