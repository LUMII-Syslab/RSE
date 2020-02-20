"""
Adapted from: https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/musicnet/scripts/resample.py

Instructions:
wget https://homes.cs.washington.edu/~thickstn/media/musicnet.npz
python3 -u resample.py musicnet.npz musicnet_11khz.npz 44100 11000
"""

from __future__ import print_function

import argparse

import numpy
from intervaltree import Interval, IntervalTree
from resampy import resample


def resample_musicnet(file_in, file_out, frame_rate, frame_rate_out):
    ratio = frame_rate_out / float(frame_rate)
    print('.. resampling {} ({}Hz) into {} ({}Hz)'.format(
        file_in, frame_rate, file_out, frame_rate_out))
    print('.. sampling with ratio {}'.format(ratio))

    resampled_data = {}
    with open(file_in, 'rb') as f_in:
      
        np_load_old = numpy.load
        numpy.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        data_in = numpy.load(file_in, encoding='latin1')
        numpy.load = np_load_old

        n_files = len(data_in.keys())
        for i, key in enumerate(data_in):
            print('.. aggregating {} ({} / {})'.format(key, i, n_files))
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

        print('.. saving output')
        with open(file_out, 'wb') as f_out:
            numpy.savez(f_out, **resampled_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_in')
    parser.add_argument('file_out')
    parser.add_argument('frame_rate', type=int)
    parser.add_argument('frame_rate_out', type=int)

    resample_musicnet(**parser.parse_args().__dict__)
