import os
import glob
import shutil
import tarfile
import urllib.request
import scipy.misc as misc

url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
dirnames = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

def download_and_uncompress_tarball(tarball_url, dataset_dir):
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath)
    tarfile.open(filepath, "r:gz").extractall(dataset_dir)


def download_and_convert(data_root):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if not os.path.exists(os.path.join(data_root, "flower_photos")):
        print("[!] Downloading images...")
        download_and_uncompress_tarball(url, data_root)

    data_dir = os.path.join(data_root, "flower")
    
    if not os.path.exists(os.path.join(data_dir, "train")):
        os.makedirs(os.path.join(data_dir, "train"))
    if not os.path.exists(os.path.join(data_dir, "test")):
        os.makedirs(os.path.join(data_dir, "test"))
    
    print("[!] Converting images...")
    for dirname in dirnames:
        paths = glob.glob(os.path.join(
            data_root, "flower_photos", dirname, "*.jpg"))
        
        # training data
        for path in paths[:-10]:
            new_path = os.path.join(data_dir, "train",
                "{}_{}".format(dirname, path.split("/")[-1]))

            im = misc.imread(path)
            misc.imsave(new_path, im)

        # test data
        for path in paths[-10:]:
            new_path = os.path.join(data_dir, "test",
                "{}_{}".format(dirname, path.split("/")[-1]))

            im = misc.imread(path)
            misc.imsave(new_path, im)
