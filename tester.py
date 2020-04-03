"""For testing the model after training is finished"""

import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

import config as cnf
import data_feeder
import data_utils as data_gen
from RSE_model import DNGPU

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use cpu only, to be able running in parallel with training
os.environ["CUDA_VISIBLE_DEVICES"] = cnf.gpu_instance
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

ATTEMPTS = 10
BATCH_SIZE = ATTEMPTS


def data_set_test(test_length):
    for input_x, target_y in data_gen.test_set[cnf.task][test_length]:

        batch_xs = []
        batch_ys = []
        for _ in range(ATTEMPTS):
            xs, ys = data_gen.add_padding(input_x, target_y, test_length)

            batch_xs.append(xs)
            batch_ys.append(ys)

        yield batch_xs, batch_ys, target_y


def prepare_data_for_test():
    data_gen.init()
    task = data_gen.find_data_task(cnf.task)
    task.prepare_test_data()
    data_gen.collect_bins()
    data_gen.print_bin_usage()


def print_words(real_in, reference, hypothesis):
    print("--------------------------------")
    print("INPUT: ", " ".join(real_in))
    answers = [pos for pos, val in enumerate(reference) if val == 2]
    print("REFERENCE:", " ".join(map(str, answers)), "REFERENCE_WORD:", real_in[answers[0]] if answers else "")
    print("HYPOTHESIS_INDEX:", hypothesis, "HYPOTHESIS_WORD:", real_in[hypothesis])


def run_test():
    answers = {bin_len: 0 for bin_len in cnf.bins}
    total_lines = {bin_len: 0 for bin_len in cnf.bins}
    prepare_data_for_test()

    for test_length in cnf.bins:

        with tf.Graph().as_default():
            tester = create_tester(test_length)

            with tf.Session(config=tf.ConfigProto()) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, cnf.model_file)

                for input_batch, target_batch, target_y in data_set_test(test_length):
                    acc1, test_result, _ = tester.get_accuracy(sess, [input_batch], [target_batch])

                    results = {}
                    for result_val, target in zip(test_result, target_batch):

                        if target[result_val] == 0:  # If shows on padding ignore this try
                            continue

                        real_position = 0
                        for pos in range(result_val):
                            if target[pos] != 0:
                                real_position += 1

                        result_val = real_position

                        if result_val not in results:
                            results[result_val] = 1
                        else:
                            results[result_val] += 1

                    answer_val = None
                    answer_frequency = 0

                    for val, frequency in results.items():
                        if frequency >= answer_frequency:
                            answer_val = val
                            answer_frequency = frequency

                    target_val = target_y
                    if answer_val and answer_val < len(target_val) and target_val[answer_val] == 2:
                        answers[test_length] += 1
                    total_lines[test_length] += 1

    for bin_len in cnf.bins:
        print("\n----------------- TEST OVERVIEW FOR LEN {len}-----------------".format(len=bin_len))
        print("Correct answers:", answers[bin_len])
        print("Total examples:", total_lines[bin_len])
        print("Accuracy:", answers[bin_len] / total_lines[bin_len])

    print("\n----------------- TEST OVERVIEW (TOTAL) -----------------")
    correct_v = sum([v for _, v in answers.items()])
    print("Correct answers:", correct_v)
    total_v = sum([v for _, v in total_lines.items()])
    print("Total examples:", total_v)
    print("Accuracy:", correct_v / total_v)


def run_test_musicnet():
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
                    labels_pre = np.array(batch_ys[0])[:,
                                 stride_labels * (n_frames // 2):stride_labels * (n_frames // 2) + 128]
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
                # print("Sampled predictions with labels:")
                # print("Predictions 1:", predictions[0:128])
                # print("Labels 1:", labels[0:128])
                # print("Predictions 2:", predictions[128:256])
                # print("Labels 2:", labels[128:256], "\n")


def correct_answers_in_batch(target_batch, result_batch) -> int:
    correct_answers = 0

    for result_val, target_val in zip(result_batch, target_batch):
        if result_val < len(target_val) and target_val[result_val] == 2:
            correct_answers += 1

    return correct_answers


def create_tester(test_length):
    learner = DNGPU(cnf.n_hidden, cnf.bins, cnf.n_input, [BATCH_SIZE], cnf.n_output, cnf.dropout_keep_prob,
                    create_translation_model=cnf.task in cnf.language_tasks, use_two_gpus=cnf.use_two_gpus)
    learner.create_test_graph(test_length)
    return learner


if __name__ == '__main__':
    if cnf.task == "musicnet":
        run_test_musicnet()
    else:
        run_test()
