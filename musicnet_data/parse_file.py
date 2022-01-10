"""
Adapted from: https://github.com/muqiaoy/dl_signal/blob/master/music/parse_file.py
This file creates .npy dataset parts from resampled musicnet
Instructions:
    First run the resample.py file. Then:
    python3 -u parse_file.py
"""

import argparse
import numpy as np
from numpy.lib.format import open_memmap
from scipy.fft import rfft  # real fast Fourier transform

n_features = 8192  # number of features (the window size)
do_fourier_transform = True
fourier_multiplier = 2  # how many times longer is fourier window

sampling_rate = 11000  # samples/second
note_types = 128  # number of distinct notes
stride_train = 512  # samples between windows
stride_test = 128  # stride in test set
data_type = 'float32'

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
data = np.load(open('musicnet_11khz.npz', 'rb'), encoding='latin1', mmap_mode='r')
np.load = np_load_old

# split the dataset into train, validation and test
test_IDs = ['2303', '2382', '1819']
validation_IDs = ['2131', '2384', '1792', '2514', '2567', '1876']
train_IDs = [ID for ID in data.files if ID not in (test_IDs + validation_IDs)]


def create_set(recording_IDs, stride, mode, filename=None):
    """Create a set of input - label pairs."""

    str_fourier = f"fourier{fourier_multiplier}" if do_fourier_transform else "raw"
    if filename is None: filename = f"musicnet_{str_fourier}_{mode}_{n_features}.npy"

    label_indices = []  # list of the indices of the labels within a window. E.g. [128, 256, 384] for n_features=512
    stride_labels = stride_test  # stride for labels and notes_test is the same
    labels_per_window = (n_features*fourier_multiplier) // stride_labels - 1  # -1 to exclude both edges of the segment
    for label_nr in range(labels_per_window):
        label_indices += [stride_labels * (label_nr + 1)]

    n_recordings = len(recording_IDs)
    window_counts = []  # list of the number of windows for each recording. [7861, 10000, ...]
    recording_start_indices = []  # list of on what window each recording starts. [0, 7860, 17860, ...]
    for recording_nr in range(n_recordings):
        x, _ = data[recording_IDs[recording_nr]]
        # windows can start from the first second until n_features from the end:
        usable_recording_length = (len(x) - sampling_rate - (n_features*fourier_multiplier))
        n_windows = (usable_recording_length // stride) + 1  # +1 to include both edges of the interval
        recording_start_indices += [sum(window_counts)]
        window_counts += [n_windows]

    xy_set = open_memmap(filename, mode='w+', dtype=data_type, shape=(sum(window_counts), 2, n_features))
    for recording_nr in range(n_recordings):
        print(f"Preparing {mode} recording {recording_nr+1}/{n_recordings}", flush=True)
        x, y = data[recording_IDs[recording_nr]]
        x_recording = np.empty([window_counts[recording_nr], n_features], dtype=data_type)
        y_recording = np.zeros([window_counts[recording_nr], (n_features*fourier_multiplier)], dtype=data_type)  # zeros for padding
        y_recording[:, :note_types * labels_per_window] = 1  # 1 for the non padded parts (1 means note not played)
        for window_nr in range(window_counts[recording_nr]):
            window_start = sampling_rate + window_nr * stride  # start from one second to give us some wiggle room
            window = np.array(x[window_start:window_start + (n_features*fourier_multiplier)], dtype=data_type)
            if do_fourier_transform:
                transformed_window = rfft(window)[0:(n_features // 2)]  # remove the last frequency to have 2^n size
                x_recording[window_nr] = transformed_window.view(dtype=data_type)  # [r1, i1, r2, i2, ...]
            else:
                x_recording[window_nr] = window
            for count, label_position in enumerate(label_indices):
                for label in y[window_start + label_position]:
                    y_recording[window_nr, note_types * count + label.data[1]] = 2  # (2 means note played)
        # crop to middle labels so that label size is n_features
        y_recording = y_recording[:, (n_features*fourier_multiplier)//2-n_features//2 : (n_features*fourier_multiplier)//2-n_features//2 + n_features]
        xy_recording = np.stack((x_recording, y_recording), axis=1)  # makes it [[[features],[labels]],..]
        xy_set[recording_start_indices[recording_nr]:recording_start_indices[recording_nr]+window_counts[recording_nr], :, :] = xy_recording


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    if args.filename:
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        data = np.load(open(args.filename, 'rb'), encoding='latin1', mmap_mode='r')
        np.load = np_load_old
        print("Creating", f"{args.filename[:-10]}.npy")
        create_set(recording_IDs=['0'], stride=stride_test, mode="test", filename=f"{args.filename[:-10]}.npy")
    else:
        create_set(recording_IDs=test_IDs, stride=stride_test, mode="test")
        create_set(recording_IDs=validation_IDs, stride=stride_test, mode="validation")
        create_set(recording_IDs=train_IDs, stride=stride_train, mode="train")
