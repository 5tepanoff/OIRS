from os import listdir, rmdir, path, rename, makedirs, stat
from os.path import isfile, join
import tarfile
import urllib.request

pokemons_url = "https://github.com/SVolkoff/pokemon-classifier"
pokemons_name = "pokemons.tgz"
train_dir = "data"

class_names = ["bulbasaur", "charmander", "mewtwo", "pikachu", "squirtle"]

CLASS_SIZE = 250

def download_dataset(filename, url, work_dir):
    if not path.exists(filename):
        print("[INFO] Downloading pokemons dataset....")

        filename, _ = urllib.request.urlretrieve(url + filename, filename)
        print("[INFO] Succesfully downloaded " + filename + " " + str(stat(filename).st_size) + " bytes.")

        tar = tarfile.open(filename)
        tar.extractall(path=work_dir, members=tar)
        tar.close()
        print("[INFO] Dataset extracted successfully.")


if __name__ == '__main__':
    if not path.exists(train_dir):
        makedirs(train_dir)

    download_dataset(pokemons_name, pokemons_url, train_dir)

    images_path = join(train_dir, 'jpg')
    images = [f for f in listdir(images_path) if isfile(join(images_path, f)) and f.endswith(".jpg")]

    class_images = [images[d:d + CLASS_SIZE] for d in range(0, len(images), CLASS_SIZE)]

    for flower_class, images in zip(class_names, class_images):
        current_path = path.join(train_dir, 'train', flower_class)

        if not path.exists(current_path):
            makedirs(current_path)

        for file in images:
            original_path = join(train_dir, 'jpg', file)
            class_path = join(train_dir, 'train', flower_class, file)
            rename(original_path, class_path)

