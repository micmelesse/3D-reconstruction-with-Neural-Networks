import os


def download_dataset(link):
    download_folder = os.path.splitext(os.path.basename(link))[0]
    archive = download_folder + ".tgz"

    if not os.path.isfile(archive):
        os.system('wget -c {0}'.format(link))

    os.system("tar -xvzf {0}".format(archive))
    os.system("mv {0} ./data".format(download_folder))
    os.system("rm -f {0}".format(archive))
    return download_folder


if __name__ == '__main__':
    LABEL_LINK = 'ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz'
    DATA_LINK = "ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz"

    LABEL_DIR = download_dataset(LABEL_LINK)
    DATA_DIR = download_dataset(DATA_LINK)
