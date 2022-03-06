"""Code for loading the parsed MusicNet dataset"""

import os
import numpy as np
from subprocess import run

import config as cnf
import data_utils
from language.utils import LanguageTask

str_fourier = f"fourier{cnf.musicnet_fourier_multiplier}" if cnf.musicnet_do_fourier_transform else "raw"
MUSICNET_TRAIN = f"musicnet_data/musicnet_{str_fourier}_train_{cnf.musicnet_file_window_size}.npy"
MUSICNET_VALIDATION = f"musicnet_data/musicnet_{str_fourier}_validation_{cnf.musicnet_file_window_size}.npy"
MUSICNET_TEST = f"musicnet_data/musicnet_{str_fourier}_test_{cnf.musicnet_file_window_size}.npy"


class Musicnet(LanguageTask):
    def __init__(self) -> None:
        self.window_size = cnf.musicnet_window_size  # how long is one input sequence (e.g. 2048)
        self.mmap_count = cnf.musicnet_mmap_count  # how many inputs there are in mmap partial load
        self.training_set = []
        self.validation_set = []
        self.testing_set = []

    def crop(self, xy_set):
        """Crop data to a smaller sized window."""
        midpoint = cnf.musicnet_file_window_size // 2
        half_window = self.window_size // 2
        result = xy_set[:, :, midpoint - half_window:midpoint + half_window]
        return list(result)  # returns [[[inputs1],[labels1]],..] (,2,window_size)

    def load_training_dataset(self):
        loaded = np.load(MUSICNET_TRAIN)
        self.training_set = self.crop(loaded)
        del loaded

    def sample_training_dataset_mmap(self):
        n_sample_locations = self.mmap_count
        loaded = np.load(MUSICNET_TRAIN, mmap_mode='r')
        mmap_window = self.mmap_count // n_sample_locations
        indices = np.random.randint(low=0, high=len(loaded) - mmap_window, size=n_sample_locations)
        training_set_tmp = []
        for i in range(n_sample_locations):
            training_set_tmp += list(loaded[indices[i]:indices[i] + mmap_window])
        self.training_set = self.crop(np.array(training_set_tmp, dtype='float32'))
        del loaded

    def prepare_data(self):
        print("Loading the dataset", flush=True)
        self.prepare_train_data()
        self.prepare_validation_data()
        data_utils.reset_counters()

    def prepare_train_data(self):
        if cnf.musicnet_subset:
            self.sample_training_dataset_mmap()
        else:
            self.load_training_dataset()
        data_utils.train_set["musicnet"][self.window_size] = self.training_set

    def prepare_validation_data(self):
        loaded = np.load(MUSICNET_VALIDATION)
        self.validation_set = self.crop(loaded)
        del loaded
        data_utils.test_set["musicnet"][self.window_size] = self.validation_set

    def prepare_test_data(self):
        loaded = np.load(MUSICNET_TEST)
        self.testing_set = self.crop(loaded)
        del loaded
        data_utils.test_set["musicnet"][self.window_size] = self.testing_set

    def prepare_inference_data(self, inference_file_path):
        loaded = np.load(inference_file_path)
        self.testing_set = self.crop(loaded)
        del loaded
        data_utils.test_set["musicnet"][self.window_size] = self.testing_set
