import os


def download_dataset(link):
    download_folder = os.path.splitext(os.path.basename(link))[0]
    archive = download_folder + ".tgz"

    if not os.path.isfile(archive):
        os.system('wget -c {0}'.format(link))

    os.system("tar -xvzf {0}".format(archive))
    os.rename(download_folder, "data/{}".format(download_dataset))
    os.system("rm -f {0}".format(archive))


def main():
    LABEL_LINK = 'ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz'
    DATA_LINK = "ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz"

    if not os.path.isdir("data/ShapeNetVox32"):
        download_dataset(LABEL_LINK)

    if not os.path.isdir("data/ShapeNetRendering"):
        download_dataset(DATA_LINK)
