import os
import dataset


def _download(_link):
    _dir = os.path.splitext(os.path.basename(_link))[0]
    _archive = _dir + ".tgz"

    if not os.path.isdir(_dir) and not os.path.isfile(_archive):
        os.system('wget -c {0}'.format(_link))
        os.system("tar -xvzf {0}".format(_archive))
    elif not os.path.isdir(_dir) and os.path.isfile(_archive):
        os.system("tar -xvzf {0}".format(_archive))
    return _dir


if __name__ == '__main__':
    LABEL_LINK = 'ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz'
    DATA_LINK = "ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz"

    LABEL_DIR = _download(LABEL_LINK)
    DATA_DIR = _download(DATA_LINK)
    dataset.write_path_csv(DATA_DIR, LABEL_DIR)
