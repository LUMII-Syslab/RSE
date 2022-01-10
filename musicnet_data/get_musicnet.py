"""
Adapted from https://github.com/jthickstun/pytorch_musicnet/blob/master/musicnet.py
Downloads the musicnet dataset; extracts it; creates npz from wav and csv; resamples to 11khz
Instructions:
    python3 get_musicnet.py
"""

import os
from subprocess import call, run
from glob import glob
import errno
import csv
import numpy as np
from scipy.io import wavfile
from intervaltree import IntervalTree

from resample import resample_musicnet


dir_path = os.path.dirname(os.path.realpath(__file__))  # path of the directory this file is in
raw_folder_path = os.path.join(dir_path, 'raw_musicnet')
url = 'https://zenodo.org/record/5120004/files/musicnet.tar.gz'


def download():
    """Download the MusicNet data if it doesn't exist already"""
    try:
        os.makedirs(raw_folder_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    from six.moves import urllib
    filename = 'musicnet.tar.gz'
    file_path = os.path.join(raw_folder_path, filename)
    if not os.path.exists(file_path):
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            # stream the download to disk (it might not fit in memory!)
            while True:
                chunk = data.read(16 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    extracted_folders = ['train_data', 'train_labels', 'test_data', 'test_labels']
    if not all(map(lambda f: os.path.exists(os.path.join(raw_folder_path, f)), extracted_folders)):
        print('Extracting ' + filename)
        if call(["tar", "-xf", file_path, '-C', raw_folder_path, '--strip', '1']) != 0:
            raise OSError("Failed tarball extraction")


def process_dataset():
    print("Processing dataset")
    dataset = {}
    items = [y for x in os.walk(raw_folder_path) for y in glob(os.path.join(x[0], '*.csv'))]
    item_nr = 1
    for item in items:  # iterate over all recording IDs
        if not item.endswith('.csv'): continue
        print(f"Processing recording {item_nr}/{len(items)}")
        item_nr += 1
        uid = str(item[:-4][-4:])  # recording ID
        if len(uid) != 4: assert False
        wav_filename = item[:-(7+len(uid)+4)]+f"data/{uid}.wav"
        _, input_audio = wavfile.read(wav_filename)  # retrieve input
        with open(item, 'r') as f:  # retrieve label
            label_tree = IntervalTree()
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:
                start_time = int(label['start_time'])
                end_time = int(label['end_time'])
                instrument = int(label['instrument'])
                note = int(label['note'])
                start_beat = float(label['start_beat'])
                end_beat = float(label['end_beat'])
                note_value = label['note_value']
                label_tree[start_time:end_time] = (instrument,note,start_beat,end_beat,note_value)
        dataset[uid] = [input_audio, label_tree]
    with open(os.path.join(dir_path, 'musicnet.npz'), 'wb') as f_out:
        np.savez(f_out, **dataset)


if __name__ == "__main__":
    if not os.path.exists(f"musicnet_11khz.npz"):
        if not os.path.exists(f"musicnet.npz"):
            download()
            process_dataset()
            run(["rm", '-r', os.path.join(dir_path, 'musicnet.tar.gz'), raw_folder_path])  # remove temporary files
        resample_musicnet("musinet.npz", "musicnet_11khz.npz", 44100, 11000)  # resample to 11khz
        run(["rm", os.path.join(dir_path, 'musinet.npz')])  # remove a temporary file
