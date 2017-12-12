def download_dataset(dataset_name="ShapeNet"):
    print("[download_dataset]")
    if(dataset_name is "ShapeNet"):
        f = urlopen(
            "http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetSem.v0/README.txt")

        shapenet_urls = re.findall('https.*', (f.read()).decode("utf-8"))
        shapenet_metadata = []
        for s in shapenet_urls:
            shapenet_metadata.append(pd.read_csv(s))

        index_metadata = 0
        id_list = (shapenet_metadata[index_metadata])["wnsynset"]
        for i, id in enumerate(id_list):
            id = 
            print(id)
            download_link = "http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/{}.zip".format(
                id)
            #urlretrieve(download_link, "./data/{}.zip" + id)
            #sys.exit()

def extract_archives(archive_link):
    im_path, im_arc = None, None
   # archive_url = requests.get(archive_link, stream=True)
   # print(archive_url.headers)
    sys.exit()

    cur_dir = os.listdir()
    if(im_path not in cur_dir):
        if(im_arc not in cur_dir):
            tarfile.open(im_path + '.tar').extractall()