"""Gets an array of predictions and labels on MusicNet test set"""

import os

import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score

import data_feeder
import config as cnf
import data_utils as data_gen
from RSE_model import RSE

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use cpu only, to be able running in parallel with training
os.environ["CUDA_VISIBLE_DEVICES"] = cnf.gpu_instance
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
BATCH_SIZE = 32


def prepare_data_for_test():
    data_gen.init()
    task = data_gen.find_data_task(cnf.task)
    task.prepare_visualisation_data()
    data_gen.collect_bins()
    data_gen.print_bin_usage()


def run_visualiser_musicnet():
    global BATCH_SIZE
    BATCH_SIZE = cnf.batch_size
    prepare_data_for_test()
    data_gen.test_counters = np.zeros(cnf.bin_max_len, dtype=np.int32)  # reset test counters
    data_supplier = data_feeder.create_data_supplier()

    for test_length in cnf.bins:

        with tf.Graph().as_default():
            tester = create_tester(test_length)

            with tf.Session(config=tf.ConfigProto()) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, cnf.model_file)

                predictions = []
                labels = []
                full_test_len = len(data_gen.test_set[cnf.task][test_length])
                # rounds it up to the next size divisible by batch size:
                n_test_inputs = (full_test_len // BATCH_SIZE) * BATCH_SIZE + BATCH_SIZE
                print("Testing on {} test inputs: ".format(n_test_inputs), end="")
                threshold = 0
                for i in range(n_test_inputs // BATCH_SIZE):
                    if i > threshold:
                        print("{},".format(i * BATCH_SIZE), end="", flush=True)
                        threshold += 1000 // BATCH_SIZE
                    batch_xs, batch_ys = data_supplier.supply_test_data(test_length, BATCH_SIZE)
                    pred_flat = (tester.get_result(sess, batch_xs, batch_ys)).flatten()

                    stride_labels = 128
                    n_frames = cnf.musicnet_window_size // stride_labels - 1
                    labels_pre = np.array(batch_ys[0])[:, stride_labels * (n_frames // 2):stride_labels * (n_frames // 2) + 128]
                    labels_flat = (labels_pre - 1).flatten()  # gets 0/1 labels on 128 notes
                    predictions += list(pred_flat)
                    labels += list(labels_flat)

                predictions = np.array(predictions)
                labels = np.array(labels)
                n_overshoot = n_test_inputs - full_test_len  # inputs more than test len
                if n_overshoot > 0:  # removes duplicates
                    predictions = predictions[:-(128 * n_overshoot)]
                    labels = labels[:-(128 * n_overshoot)]

                avg_prec_score = average_precision_score(labels, predictions)
                print("\n")
                print("Cutting {} input duplicates".format(n_overshoot))
                print("Done testing on all {} test inputs".format(len(labels) / 128))
                print("AVERAGE PRECISION SCORE on all test data = {0:.7f}".format(avg_prec_score))

                # preparing the visualisation data:
                t_probe_start = 128 * 0
                t_probe_len = len(labels)
                predictions = predictions[t_probe_start:t_probe_start + t_probe_len]
                labels = labels[t_probe_start:t_probe_start + t_probe_len]
                predictions = np.reshape(predictions, (-1, 128))
                labels = np.reshape(labels, (-1, 128))
                predictions = np.transpose(predictions, axes=(1, 0))
                labels = np.transpose(labels, axes=(1, 0))
                np.save("visualisation.npy", np.array([predictions, labels]))


def create_tester(test_length):
    learner = RSE(cnf.n_hidden, cnf.bins, cnf.n_input, [BATCH_SIZE], cnf.n_output, cnf.dropout_keep_prob,
                  create_translation_model=cnf.task in cnf.language_tasks, use_two_gpus=cnf.use_two_gpus)
    learner.create_test_graph(test_length)
    return learner


if __name__ == '__main__':
    run_visualiser_musicnet()
