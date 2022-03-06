"""
If you have a trained model, this file will run inference on the wav file and create a MIDI file
Instructions:
    Place a wav file <filename>.wav in the musicnet_data folder and run:
    python3 transcribe.py <filename>.wav
"""

import os
import argparse
import numpy as np
from subprocess import run
from scipy.io import wavfile
from intervaltree import IntervalTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mido

from resample import resample_musicnet


def wav_to_npz(filename_wav):
    """Converts wav file to npz file"""
    samplerate, data = wavfile.read(filename_wav)
    if max(data) > 2:
        print("The wav file should be normed to [-1,1]. Attempting to normalize")
        normed_data = data / (np.linalg.norm(data) + 10e-6)  # converts to interval [-1, 1]
    else:
        normed_data = data
    padding = np.zeros(samplerate, dtype=np.float32)
    padded_data = np.concatenate((padding, normed_data), axis=0)  # add 1 second for parser to cut off
    padded_data = padded_data.astype('float32')
    data_dict = {}
    data_dict['0'] = [padded_data, IntervalTree()]  # assumes that labels are unknown and saves with empty labels
    np.savez(f"{filename}.npz", **data_dict)
    return samplerate


def process_notes(predictions, labels):
    processing_length = len(predictions) // 128
    test_x = predictions[0:128 * processing_length].reshape((processing_length, 128)).T
    test_y = labels[0:128 * processing_length].reshape((processing_length, 128)).T
    processed_labels = np.zeros((processing_length, 128))
    processed_predictions = np.zeros((processing_length, 128))
    for window in range(processed_labels.shape[0]):
        for count, prediction in enumerate(test_x[:, window]):
            processed_predictions[window, count] = prediction
        for count, label in enumerate(test_y[:, window]):
            processed_labels[window, count] = label
    processed_labels = processed_labels.T
    processed_predictions = processed_predictions.T
    return processed_predictions, processed_labels


def binarize_predictions(predictions, threshold, velocity=64):
    """Converts predictions to ones and zeros. Each value is set to 1 if it was larger than threshold, 0 otherwise"""
    n_notes = predictions.shape[0]
    n_timesteps = predictions.shape[1]
    binarized_predictions = np.zeros((n_notes, n_timesteps), dtype=np.int)
    for t in range(n_timesteps):
        for n in range(n_notes):
            if predictions[n][t] > threshold:
                binarized_predictions[n][t] = velocity
    return binarized_predictions


def prediction_smoothing(predictions, kernel_size=4, velocity=64):
    """Runs a smoothing convolution to avoid rapid on/off switching of the resulting notes"""
    n_notes = predictions.shape[0]
    n_timesteps = predictions.shape[1]
    smooth_predictions = np.zeros((n_notes, n_timesteps), dtype=np.int)
    for t in range(kernel_size // 2, n_timesteps - kernel_size // 2):
        for n in range(n_notes):
            if np.average(predictions[n][(t - kernel_size // 2):(t + kernel_size // 2)]) > velocity//2:
                smooth_predictions[n][t] = velocity
    return smooth_predictions


def array2midi(arr, tempo=500000):
    """Converts a numpy array to a MIDI file"""
    # Adapted from: https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    new_arr = np.concatenate([np.array([[0] * 128]), np.array(arr)], axis=0)
    changes = new_arr[1:] - new_arr[:-1]
    midi_file = mido.MidiFile()  # create a midi file with an empty track
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    last_time = 0
    for ch in changes:  # add difference in the empty track
        if set(ch) == {0}:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_vol = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_vol):
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_on', note=n, velocity=v, time=new_time))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=n, velocity=0, time=new_time))
                first_ = False
            last_time = 0
    return midi_file


def visualise_notes(predictions, labels, filename):
    """Saves a visualisation of note predictions and labels"""
    image_length = min(640, len(predictions) // 128)  # 640 is a decent image length
    test_x = predictions[0:128 * image_length].reshape((image_length, 128)).T
    test_y = labels[0:128 * image_length].reshape((image_length, 128)).T

    # create the image matrix
    image_labels = np.zeros((image_length, 3 * 128))
    image_predictions = np.zeros((image_length, 3 * 128))
    for window in range(image_labels.shape[0]):
        for count, prediction in enumerate(test_x[:, window]):
            image_predictions[window, 3 * count + 0] = prediction
            image_predictions[window, 3 * count + 1] = prediction
            image_predictions[window, 3 * count + 2] = prediction
        for count, label in enumerate(test_y[:, window]):
            image_labels[window, 3 * count + 1] = label
    image_matrix = image_predictions.T + 2*image_labels.T  # scaled to match the color map

    # define a color map
    c = mcolors.ColorConverter().to_rgb
    seq = [c('white'), c('black'), 0.33, c('black'), c('#bb0000'), 0.66, c('#bb0000'), c('#00bb00')]
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    wbrg = mcolors.LinearSegmentedColormap('CustomMap', cdict)  # white, black, red, green

    # plot the image
    fig = plt.figure(figsize=(640 / 5, 150 / 2), dpi=20)  # values to have pixels at integer positions
    plt.imshow(image_matrix, aspect='auto', cmap=wbrg, vmin=0, vmax=3)
    plt.gca().invert_yaxis()
    plt.ylim(3 * 40, 3 * 90)  # show notes from 40 to 90
    plt.axis('off')

    # save the image
    plt.savefig(filename, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualise_binarized_notes(image_matrix, filename):
    """Saves a visualisation of binarized note predictions and labels"""
    c = mcolors.ColorConverter().to_rgb  # define a color map
    seq = [c('white'), c('black'), 0.33, c('black'), c('#bb0000'), 0.66, c('#bb0000'), c('#00bb00')]
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    wbrg = mcolors.LinearSegmentedColormap('CustomMap', cdict)  # white, black, red, green
    fig = plt.figure(figsize=(640 / 5, 150 / 2), dpi=20)  # values to have pixels at integer positions
    plt.imshow(image_matrix, aspect='auto', cmap=wbrg, vmin=0, vmax=3)
    plt.ylim(40, 90)  # show notes from 40 to 90
    plt.axis('off')
    plt.savefig(filename, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    threshold = 0.5  # threshold for binarizing predictions
    velocity = 64  # MIDI volume
    visualise_predictions_and_labels = True
    visualise_binarized_predictions_and_labels = False

    parser = argparse.ArgumentParser()
    parser.add_argument('filename_wav')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))  # path of the directory this file is in
    filename = args.filename_wav[:-4]
    inference_file_path = os.path.join(dir_path, f"{filename}.npy")
    results_file_path = os.path.join(dir_path, f"{filename}_results.npy")

    if not os.path.exists(results_file_path):
        samplerate = wav_to_npz(args.filename_wav)  # wav -> npz
        resample_musicnet(f"{filename}.npz", f"{filename}_11khz.npz", samplerate, 11000)  # resample to 11khz
        run(["python3", "parse_file.py", "--filename", f"{filename}_11khz.npz"])  # parse file so it can be processed by the model
        tester_path = f"{os.path.join(dir_path, '..', 'tester.py')}"
        run(["python3", tester_path, inference_file_path, f"{filename}_results.npy"])  # run inference
        run(["rm", f"{filename}.npz", f"{filename}_11khz.npz", f"{filename}.npy"])  # remove temporary files
    else:
        print(f"{results_file_path} already exists. Proceeding with the existing version")

    with open(results_file_path, 'rb') as f_in:
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        results = np.load(results_file_path, encoding='latin1')
        np.load = np_load_old
    predictions, labels = results

    processed_predictions, processed_labels = process_notes(predictions, labels)
    processed_predictions = binarize_predictions(processed_predictions, threshold=threshold, velocity=velocity)
    processed_predictions = prediction_smoothing(processed_predictions, velocity=velocity)

    print("Converting predictions to MIDI")
    midi_notes = np.repeat(processed_predictions.T, 11, axis=0)  # adjust tempo
    midi_file = array2midi(midi_notes, tempo=int(500000 / (88 / 90)))  # adjust tempo
    midi_file.save(f"{filename}.mid")

    if visualise_predictions_and_labels:
        visualise_notes(predictions, labels, f"{filename}_visualisation.png")
    if visualise_binarized_predictions_and_labels:
        processed_labels = binarize_predictions(processed_labels, threshold=0.5, velocity=velocity)
        visualise_binarized_notes(
            processed_predictions[:, :640] * 1 / (velocity + 1) + processed_labels[:, :640] * 2 / velocity,
            f"{filename}_visualisation_binarized.png")
