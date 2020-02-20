import io
import os
import pickle
import random
import tarfile

import numpy as np
import tensorflow as tf
from six.moves import urllib

import config as cnf


def extract_tar(tar_path, location):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(location)


def download(directory, filename, url):
    """Download filename from url unless it's already in directory."""
    if not tf.gfile.Exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)

    path = os.path.join(directory, filename)

    if not tf.gfile.Exists(path):
        print("Downloading %s to %s" % (url, path))
        path, _ = urllib.request.urlretrieve(url, path)
        statinfo = os.stat(path)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes")

    return path


def prepare_embeddings():
    file_in = io.open(cnf.embedding_file, 'r', encoding="utf-8", newline='\n', errors='ignore')

    n, d = list(map(int, file_in.readline().split(' ')))  # Dimensions of embedding
    print("Preparing embedding with dimensions:", n, d)

    word_to_id = {"C_PAD": 0, "C_UNK": 1}
    word_to_vec = [['0' for _ in range(d)], [str(random.random()) for _ in range(d)]]

    word_id = 2

    with open(cnf.emb_vector_file, "w") as vector_file:

        for line in file_in:
            tokens = line.rstrip().split()
            word_to_vec.append(tokens[1:])
            word_to_id[tokens[0]] = word_id

            if word_id % 100000 == 0:
                vector_file.writelines([" ".join(word) + "\n" for word in word_to_vec])
                word_to_vec = []
                print("Done with", word_id, "word")

            word_id += 1

        vector_file.writelines([" ".join(word) + "\n" for word in word_to_vec])

    with open(cnf.emb_word_dictionary, "wb") as id_out:
        pickle.dump(word_to_id, id_out)  # For faster load times save numpy array as binary file

    print('Pickled word dictionary')

    vector_file = np.loadtxt(cnf.emb_vector_file, dtype=np.float)
    with open(cnf.emb_vector_file, "wb") as emb_file_bin:  # For faster load times save numpy array as binary file
        pickle.dump(vector_file, emb_file_bin)

    print('Pickled embedding')


class LanguageTask:

    def prepare_data(self):
        pass

    def prepare_train_data(self):
        pass

    def prepare_test_data(self):
        pass

    def detokenizer(self):
        pass
