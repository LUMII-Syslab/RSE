# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Code for training the Residual Shuffle-Exchange model."""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '20'

import tensorflow as tf
import time
from datetime import datetime
from sklearn.metrics import average_precision_score
import numpy as np
import config as cnf
import data_utils as data_gen
from RSE_model import RSE
import data_feeder


print("Running Residual Shuffle-Exchange network trainer.....")

if not cnf.use_two_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = cnf.gpu_instance
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

countList = [cnf.batch_size for x in cnf.bins]
np.set_printoptions(linewidth=2000, precision=4, suppress=True)

# prepare training and test data
max_length = cnf.bins[-1]
data_gen.init()

if cnf.task in cnf.language_tasks:
    task = data_gen.find_data_task(cnf.task)
    task.prepare_data()
    data_gen.collect_bins()
    data_gen.print_bin_usage()
else:
    for length in range(1, max_length + 1):
        n_examples = cnf.data_size
        data_gen.init_data(cnf.task, length, n_examples, cnf.n_input)
    data_gen.collect_bins()
    if len(data_gen.train_set[cnf.task][cnf.forward_max]) == 0:
        data_gen.init_data(cnf.task, cnf.forward_max, cnf.test_data_size, cnf.n_input)

data_supplier = data_feeder.create_data_supplier()


# Perform training
with tf.Graph().as_default():
    learner = RSE(cnf.n_hidden, cnf.bins, cnf.n_input, countList, cnf.n_output, cnf.dropout_keep_prob,
                  create_translation_model=cnf.task in cnf.language_tasks, use_two_gpus=cnf.use_two_gpus)
    learner.create_graph()
    learner.variable_summaries = tf.summary.merge_all()
    tf.get_variable_scope().reuse_variables()
    learner.create_test_graph(cnf.forward_max)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=cnf.tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = "{time}_{task}".format(time=current_time, task=cnf.task)
        if len(sys.argv) > 1:  # if run_name is passed as CL argument
            run_name = str(sys.argv)[1]
        output_dir = os.path.join(cnf.out_dir, run_name)
        train_writer = tf.summary.FileWriter(output_dir)

        if cnf.load_prev:
            saver1 = tf.train.Saver([var for var in tf.trainable_variables()])
            saver1.restore(sess, cnf.model_file)

        tvars = tf.trainable_variables()
        vsum = 0
        for v in tvars:
            vsum += np.product(v.get_shape().as_list())
        print("learnable parameters:", vsum / 1024 / 1024, 'M', flush=True)

        batch_xs, batch_ys = data_supplier.supply_validation_data(max_length, cnf.batch_size)
        step = 1
        loss = 0
        avgLoss = 0
        avgRegul = 0
        acc = 1
        prev_loss = [1000000] * 7
        # max_APS = 0
        start_time = time.time()
        batch_xs_long, batch_ys_long = data_supplier.supply_test_data(cnf.forward_max, cnf.batch_size)
        long_accuracy, _, _ = learner.get_accuracy(sess, batch_xs_long, batch_ys_long)
        print("Iter", 0, "time =", 0)
        print("accuracy on test length", cnf.forward_max, "=", long_accuracy)

        while step < cnf.training_iters:
            if step % cnf.display_step == 0:
                avgLoss /= cnf.display_step
                avgRegul /= cnf.display_step
                step_time = time.time() - start_time
                start_time = time.time()
                lr = learner.get_learning_rate(sess)
                if step % 10000 == 0: saver.save(sess, cnf.model_file)
                print("Iter", step, "time =", step_time, "lr =", lr, 'max_loss =', loss, 'min_accuracy =', acc,
                      'avgLoss =', avgLoss)
                summaries = learner.print_loss(sess, batch_xs, batch_ys)
                train_writer.add_summary(summaries, step)

                batch_xs_long, batch_ys_long = data_supplier.supply_test_data(cnf.forward_max, cnf.batch_size)
                long_accuracy, _, test_summary = learner.get_accuracy(sess, batch_xs_long, batch_ys_long)
                train_writer.add_summary(test_summary, step)
                print("accuracy on length", cnf.forward_max, "=", long_accuracy)

                # set saturation weight proportional to average loss
                learner.set_saturation_weight(sess, avgLoss / (avgRegul + 1e-20))

                # decrease learning rate if no progress is made in 4 checkpoints
                prev_loss.append(avgLoss)
                if min(prev_loss[-3:]) > min(prev_loss[-4:]):
                    prev_loss = [1000000] * 7
                    sess.run(learner.lr_decay_op)
                loss = 0
                acc = 1
                avgLoss = 0
                avgRegul = 0

            # for musicnet:
            # --------------------------------------
            if cnf.task == "musicnet" and step % cnf.musicnet_test_step == 0:
                if cnf.musicnet_resample:
                    print("Resampling the data", flush=True)
                    task.prepare_data()  # resampling

                n_test_inputs = cnf.musicnet_n_test_batches * cnf.batch_size
                n_trials = 2
                if step % cnf.musicnet_full_test_step == 0:  # full test
                    data_gen.test_counters = np.zeros(cnf.bin_max_len, dtype=np.int32)  # reset test counters
                    full_test_len = len(data_gen.test_set[cnf.task][cnf.forward_max])
                    # rounds it up to the next size divisible by batch size:
                    n_test_inputs = (full_test_len // cnf.batch_size) * cnf.batch_size + cnf.batch_size
                    n_trials = 1
                avg_prec_scores = [0.0] * n_trials
                for trial in range(n_trials):
                    predictions = []
                    labels = []
                    print("Testing on {} inputs".format(n_test_inputs))
                    for i in range(n_test_inputs // cnf.batch_size):
                        batch_xs_long, batch_ys_long = data_supplier.supply_test_data(cnf.forward_max, cnf.batch_size)
                        pred_flat = (learner.get_result(sess, batch_xs_long, batch_ys_long)).flatten()
                        stride_labels = 128
                        n_frames = cnf.musicnet_window_size // stride_labels - 1
                        labels_pre = np.array(batch_ys_long[0])[:, stride_labels*(n_frames//2):stride_labels*(n_frames//2)+128]
                        labels_flat = (labels_pre - 1).flatten()  # gets 0/1 labels on 128 notes
                        predictions += list(pred_flat)
                        labels += list(labels_flat)
                    predictions = np.array(predictions)
                    labels = np.array(labels)
                    if step % cnf.musicnet_full_test_step == 0:
                        n_overshoot = n_test_inputs - full_test_len  # inputs more than test len
                        if n_overshoot > 0:  # removes duplicates
                            predictions = predictions[:-(128 * n_overshoot)]
                            labels = labels[:-(128 * n_overshoot)]
                        print("Cutting {} input duplicates. Total inputs = {}".format(n_overshoot, len(labels) // 128))
                    avg_prec_scores[trial] = average_precision_score(labels, predictions)
                for trial in range(n_trials):
                    print("Test {} - AVERAGE PRECISION SCORE = {:.5f}".format(trial, avg_prec_scores[trial]))
                aps_test_summary = tf.compat.v1.Summary()
                avg_APS = sum(avg_prec_scores) / len(avg_prec_scores)
                diff_APS = max(avg_prec_scores) - min(avg_prec_scores)
                aps_test_summary.value.add(tag='test_APS/avg_APS', simple_value=avg_APS)
                aps_test_summary.value.add(tag='test_APS/diff_APS', simple_value=diff_APS)
                print("")
                if step % cnf.musicnet_full_test_step == 0:
                    full_APS = avg_prec_scores[0]
                    aps_test_summary.value.add(tag='test_APS/full_APS', simple_value=full_APS)
                    print("Predictions:", predictions[0:128])
                    print("Labels:", labels[0:128])
                    print("\n")
                train_writer.add_summary(aps_test_summary, step)
                train_writer.flush()
            # -------------------------------

            batch_xs, batch_ys = data_supplier.supply_training_data(max_length, cnf.batch_size)
            loss1, acc1, perItemCost, costList, regul1 = learner.train(sess, batch_xs, batch_ys)
            avgLoss += loss1
            avgRegul += regul1

            loss = max(loss, loss1)
            acc = min(acc, acc1)
            step += 1

        print("Optimization Finished!")
        saver.save(sess, cnf.model_file)

# test generalization to longer examples
test_length = 8
data_gen.init()

while test_length < cnf.max_test_length:
    if len(data_gen.test_set[cnf.task][test_length]) == 0:
        data_gen.init_data(cnf.task, test_length, cnf.test_data_size, cnf.n_input)

    tmp_length = test_length
    while len(data_gen.test_set[cnf.task][tmp_length]) == 0 and tmp_length > 1:
        tmp_length -= 1
        data_gen.init_data(cnf.task, tmp_length, cnf.test_data_size, cnf.n_input)

    data_gen.reset_counters()
    batchSize = 1
    if test_length < 2000: batchSize = 16
    if test_length < 800: batchSize = 128

    with tf.Graph().as_default():
        tester = RSE(cnf.n_hidden, [test_length], cnf.n_input, [batchSize], cnf.n_output, cnf.dropout_keep_prob)
        tester.create_test_graph(test_length)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=cnf.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, cnf.model_file)
            errors, seq_errors, total = 0, 0, 0
            for iter in range(cnf.test_data_size // batchSize):
                batch_xs, batch_ys = data_supplier.supply_test_data(test_length, batchSize)
                acc1, test_result, _ = tester.get_accuracy(sess, batch_xs, batch_ys)
                er, tot, seq_er = data_gen.accuracy(batch_xs[0], test_result, batch_ys[0], batchSize, 0)
                errors += er
                seq_errors += seq_er
                total += tot

            acc_real = 1.0 - float(errors) / total
            print("Testing length:", test_length, "accuracy=", acc_real, "errors =", errors, "incorrect sequences=",
                  seq_errors)
    test_length = test_length * 2
