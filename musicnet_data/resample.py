"""
Adapted from: https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/musicnet/scripts/resample.py
Instructions:
    wget https://homes.cs.washington.edu/~thickstn/media/musicnet.npz
    python3 -u resample.py musicnet.npz musicnet_11khz.npz 44100 11000
"""

from __future__ import print_function

import argparse

import numpy as np
from intervaltree import Interval, IntervalTree
from resampy import resample


def resample_musicnet(file_in, file_out, frame_rate, frame_rate_out):
    ratio = frame_rate_out / float(frame_rate)
    print(f'Resampling {file_in} ({frame_rate}Hz) into {file_out} ({frame_rate_out}Hz)')
    print(f'Sampling with ratio {ratio}')

    resampled_data = {}
    with open(file_in, 'rb') as f_in:
      
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        data_in = np.load(file_in, encoding='latin1', mmap_mode='r')
        np.load = np_load_old

        n_files = len(data_in.keys())
        for i, key in enumerate(data_in):
            print(f'Aggregating {key} ({i} / {n_files})')
            data = data_in[key]
            data[0] = resample(data[0], frame_rate, frame_rate_out)
            resampled_intervals = []
            for interval in data[1]:
                resampled_begin = int(interval.begin * ratio)
                resampled_end = int(interval.end * ratio)
                resampled_interval = Interval(
                    resampled_begin, resampled_end, interval.data)
                resampled_intervals.append(resampled_interval)
            data[1] = IntervalTree(resampled_intervals)
            resampled_data[key] = data

        print('Saving output')
        with open(file_out, 'wb') as f_out:
            np.savez(f_out, **resampled_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_in')
    parser.add_argument('file_out')
    parser.add_argument('frame_rate', type=int)
    parser.add_argument('frame_rate_out', type=int)

    resample_musicnet(**parser.parse_args().__dict__)
