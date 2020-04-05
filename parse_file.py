"""
Adapted from: https://github.com/muqiaoy/dl_signal/blob/master/music/parse_file.py
This file creates .npy dataset parts from resampled musicnet
Instructions:
* First run the resample.py file. Then:
* python3 -u parse_file.py
"""

import numpy as np

fs = 11000  # samples/second
d = 8192  # number of features
m = 128  # number of distinct notes
stride_train = 512  # samples between windows
stride_test = 128  # stride in test set
data_type = 'float32'

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
data = np.load(open('musicnet_11khz.npz', 'rb'), encoding='latin1')
np.load = np_load_old

# split the dataset into train, validation and test
test_IDs = ['2303', '2382', '1819']
validation_IDs = ['2131', '2384', '1792', '2514', '2567', '1876']
train_IDs = [ID for ID in data.files if ID not in (test_IDs + validation_IDs)]


def create_set(n_songs, stride, mode):
    """Create a set of input - label pairs."""
    xy_set = []
    for i in range(n_songs):
        print("Preparing {} song {}".format(mode, i + 1))
        if mode == "train":
            x, y = data[train_IDs[i]]
        elif mode == "validation":
            x, y = data[validation_IDs[i]]
        else:
            x, y = data[test_IDs[i]]

        n_inputs = int((len(x) - fs - d) // stride + 1)  # number of sequences in the current song
        stride_labels = stride_test
        n_frames = d // stride_labels - 1  # -1 to exclude the edges of the segment
        frame_positions = []  # positions of the labeled frames
        for i in range(n_frames):
            frame_positions += [stride_labels * (i + 1)]
        x_song = np.empty([n_inputs, d], dtype=data_type)
        y_song = np.zeros([n_inputs, d], dtype=data_type)  # zeros for padding
        y_song[:, :m * n_frames] = 1  # 1 for the non padded parts
        for j in range(n_inputs):
            s = fs + j * stride  # start from one second to give us some wiggle room for larger segments
            x_song[j] = x[s:s + d]
            for count, label_pos in enumerate(frame_positions):
                for label in y[s + label_pos]:
                    y_song[j, m * count + label.data[1]] = 2  # note played

        xy_song = np.stack((x_song, y_song), axis=1)  # makes it [[[features],[labels]],..]
        xy_set += list(xy_song)
    xy_set = np.array(xy_set, dtype=data_type)
    np.save("musicnet_{}.npy".format(mode), xy_set)


create_set(n_songs=len(train_IDs), stride=stride_train, mode="train")
create_set(n_songs=len(validation_IDs), stride=stride_test, mode="validation")
create_set(n_songs=len(test_IDs), stride=stride_test, mode="test")


# def create_song(stride, song_ID):
#     """Create a set of input - label pairs."""
#     xy_set = []
#     x, y = data[song_ID]
#
#     # n_inputs = int((len(x) - fs - d) // stride + 1)  # number of sequences in the current song
#     n_inputs = int((len(x) - d) // stride + 1)  # number of sequences in the current song
#     stride_labels = stride_test
#     n_frames = d // stride_labels - 1  # -1 to exclude the edges of the segment
#     frame_positions = []  # positions of the labeled frames
#     for i in range(n_frames):
#         frame_positions += [stride_labels * (i + 1)]
#     x_song = np.empty([n_inputs, d], dtype=data_type)
#     y_song = np.zeros([n_inputs, d], dtype=data_type)  # zeros for padding
#     y_song[:, :m * n_frames] = 1  # 1 for the non padded parts
#     for j in range(n_inputs):
#         # s = fs + j * stride  # start from one second to give us some wiggle room for larger segments
#         s = j * stride
#         x_song[j] = x[s:s + d]
#         for count, label_pos in enumerate(frame_positions):
#             for label in y[s + label_pos]:
#                 y_song[j, m * count + label.data[1]] = 2  # note played
#
#     xy_song = np.stack((x_song, y_song), axis=1)  # makes it [[[features],[labels]],..]
#     xy_set += list(xy_song)
#     xy_set = np.array(xy_set, dtype=data_type)
#     np.save("musicnet_{}.npy".format(song_ID), xy_set)
#
# for ID in test_IDs:
#     create_song(stride=stride_test, song_ID=ID)