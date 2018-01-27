import os


def check(train_dir):
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)


if __name__ == '__main__':
    TRAIN_DIRS = ["out", "config", "data", "aws"]
    for d in TRAIN_DIRS:
        check(d)
