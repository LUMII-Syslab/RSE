"""
Code for loading the parsed MusicNet dataset.
"""

import copy
from pathlib import Path

import numpy as np

import config as cnf
import data_utils
from language.utils import LanguageTask

MUSICNET_TRAIN = "/musicnet_train.npy"
MUSICNET_VALIDATION = "/musicnet_validation.npy"
MUSICNET_TEST = "/musicnet_test.npy"
INPUT_LENGTH = 8192  # non-cropped window size


class Musicnet(LanguageTask):
    def __init__(self) -> None:
        self.window_size = cnf.musicnet_window_size  # how long is one input sequence (e.g. 2048)
        self.mmap_count = cnf.musicnet_mmap_count  # how many inputs there are in mmap partial load
        self.data_dir = str(Path(cnf.musicnet_data_dir))
        self.training_set = []
        self.validation_set = []
        self.testing_set = []

    def crop(self, xy_set):
        """Crop data to a smaller sized window."""
        midpoint = INPUT_LENGTH // 2
        half_window = self.window_size // 2
        inputs = np.expand_dims(xy_set[:, 0, midpoint - half_window:midpoint + half_window], axis=1)
        labels = np.expand_dims(xy_set[:, 1, :self.window_size], axis=1)  # crops padding
        result = np.concatenate((inputs, labels), axis=1)
        return list(result)  # returns [[[inputs1],[labels1]],..] (,2,wsize)

    def load_training_dataset(self):
        loaded = np.load(str(self.data_dir) + MUSICNET_TRAIN)
        self.training_set = self.crop(loaded)
        del loaded

    def sample_training_dataset_mmap(self):
        loaded = np.load(self.data_dir + MUSICNET_TRAIN, mmap_mode='r')
        mmap_window = self.mmap_count // 100  # samples from 100 indices
        indices = np.random.randint(low=0, high=len(loaded) - mmap_window, size=100)
        training_set_tmp = []
        for i in range(100):
            training_set_tmp += list(loaded[indices[i]:indices[i] + mmap_window])
        self.training_set = self.crop(np.array(training_set_tmp, dtype='float32'))
        del loaded

    def load_training_dataset_mmap(self):
        self.training_set = []
        for i in range(10):
            loaded = np.load(self.data_dir + MUSICNET_TRAIN, mmap_mode='r')
            part_size = 1 + len(loaded) // 10  # +1 to include all data
            self.training_set += copy.deepcopy(self.crop(loaded[i * part_size:(i + 1) * part_size]))
            print("Loaded training set part {}".format(i + 1), flush=True)
            del loaded

    def prepare_data(self):
        print("Loading the dataset", flush=True)
        self.prepare_train_data()
        self.prepare_validation_data()
        data_utils.reset_counters()

    def prepare_train_data(self):
        if cnf.musicnet_mmap_load:
            self.load_training_dataset_mmap()
        elif cnf.musicnet_mmap_partial:
            self.sample_training_dataset_mmap()
        else:
            self.load_training_dataset()
        data_utils.train_set["musicnet"][self.window_size] = self.training_set

    def prepare_validation_data(self):
        loaded = np.load(self.data_dir + MUSICNET_VALIDATION)
        self.validation_set = self.crop(loaded)
        del loaded
        np.random.shuffle(self.validation_set)
        data_utils.test_set["musicnet"][self.window_size] = self.validation_set

    def prepare_test_data(self):
        loaded = np.load(self.data_dir + MUSICNET_TEST)
        self.testing_set = self.crop(loaded)
        del loaded
        data_utils.test_set["musicnet"][self.window_size] = self.testing_set

    def prepare_visualisation_data(self):
        loaded = np.load(self.data_dir + "/musicnet_visuals.npy")
        self.testing_set = self.crop(loaded)
        del loaded
        data_utils.test_set["musicnet"][self.window_size] = self.testing_set
