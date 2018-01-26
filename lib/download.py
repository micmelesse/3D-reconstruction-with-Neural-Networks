import os


def download_dataset(link):
    download_dir = os.path.splitext(os.path.basename(link))[0]
    archive = download_dir + ".tgz"

    if not os.path.isdir(download_dir) and not os.path.isfile(archive):
        os.system('wget -c {0}'.format(link))
        os.system("tar -xvzf {0}".format(archive))
        os.system("rm -f {0}".format(archive))
    elif not os.path.isdir(download_dir) and os.path.isfile(archive):
        os.system("tar -xvzf {0}".format(archive))
        os.system("rm -f {0}".format(archive))
    return download_dir


if __name__ == '__main__':
    LABEL_LINK = 'ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz'
    DATA_LINK = "ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz"

    LABEL_DIR = download_dataset(LABEL_LINK)
    DATA_DIR = download_dataset(DATA_LINK)
