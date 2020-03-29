# This code is adapted from https://github.com/tensorflow/models/tree/master/neural_gpu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for working with the data"""

import random
import sys
import time

import numpy as np
from tensorflow.python.platform import gfile

import config as cnf
import task as tasks
from language.lambada import LambadaTask
from language.musicnet import Musicnet
from language.utils import LanguageTask


def find_data_task(task: str) -> LanguageTask:
    if task == "lambada":
        return LambadaTask()
    elif task == "musicnet":
        return Musicnet()
    else:
        raise NotImplementedError("Task '{task}' not supported".format(task=task))


def get_prev_indices(n_bits):
    length = 1 << n_bits
    ptr = [-1] * length
    for k in range(1, n_bits):
        ofs = ptr.index(-1)
        step = 1 << k
        prev = -2
        while ofs < length:
            assert ptr[ofs] == -1
            ptr[ofs] = prev
            prev = ofs
            ofs += step

    return ptr


train_counters = np.zeros(cnf.bin_max_len, dtype=np.int32)
test_counters = np.zeros(cnf.bin_max_len, dtype=np.int32)


def reset_counters():
    global train_counters
    global test_counters
    train_counters = np.zeros(cnf.bin_max_len, dtype=np.int32)
    test_counters = np.zeros(cnf.bin_max_len, dtype=np.int32)


reset_counters()


def pad(length):
    for b in cnf.bins:
        if b >= length: return b
    return cnf.forward_max


train_set = {}
test_set = {}


def init(max_length=cnf.bin_max_len):
    train_set.clear()
    test_set.clear()
    for some_task in cnf.all_tasks:
        train_set[some_task] = []
        test_set[some_task] = []
        for all_max_len in range(max_length):
            train_set[some_task].append([])
            test_set[some_task].append([])


def collect_bins():
    max_length = cnf.bins[-1]
    for some_task in cnf.all_tasks:
        for L in range(max_length):
            bin_length = pad(L)
            if bin_length != L:
                cur_train = train_set[some_task]
                cur_test = test_set[some_task]
                cur_train[bin_length] += cur_train[L]
                cur_test[bin_length] += cur_test[L]
                cur_train[L] = []
                cur_test[L] = []

    # add some shorter instances to train for padding
    for some_task in cnf.all_tasks:
        for ind in range(1, len(cnf.bins)):
            small_count = len(train_set[some_task][cnf.bins[ind]]) // 20  # 5% shorter instances
            for itemNr in range(small_count):
                smaller_bin = cnf.bins[random.randint(0, ind - 1)]
                if len(train_set[some_task][smaller_bin]) > 0:
                    item = random.choice(train_set[some_task][smaller_bin])
                    train_set[some_task][cnf.bins[ind]].append(item)

    # shuffle randomly
    for some_task in cnf.all_tasks:
        for L in cnf.bins:
            random.shuffle(train_set[some_task][L])
            if not cnf.musicnet_visualise:
                random.shuffle(test_set[some_task][L])


def init_data(task, length, nbr_cases, nclass):
    init_data_1(task, length, nbr_cases, nclass, train_set)
    init_data_1(task, length, nbr_cases, nclass, test_set)


"""Data initialization."""


def init_data_1(task, length, nbr_cases, nclass, cur_set):
    cur_set[task][length] = []
    L = length
    cur_time = time.time()
    total_time = 0.0
    input_set = set()
    case_count = 0
    trials = 0

    task_gen = tasks.select_task(task, nclass)

    while case_count < nbr_cases and trials < 20:
        total_time += time.time() - cur_time
        cur_time = time.time()
        if L > cnf.bin_max_len and case_count % 100 == 1:
            print_out("  avg gen time %.4f s" % (total_time / float(case_count)))

        i, t = task_gen.input_output_pair(L)
        if len(i) == 0: break

        i_tuple = tuple(i)
        if i_tuple not in input_set:
            input_set.add(i_tuple)
            cur_set[task][len(i)].append([i, t])
            case_count += 1
            trials = 0
        else:
            trials += 1


def get_batch(max_length, batch_size, do_train, task, offset=None, preset=None):
    """Get a batch of data, training or testing."""
    inputs = []
    targets = []
    length = max_length
    if preset is None:
        if do_train:
            cur_set = train_set[task]
            counters = train_counters
        else:
            cur_set = test_set[task]
            counters = test_counters
        while not cur_set[length]:
            length -= 1
            assert length, "Bin in length {len} is empty. Expected to contain values".format(len=max_length)

    for b in range(batch_size):
        if preset is None:
            cur_ind = counters[length]
            elem = cur_set[length][cur_ind]
            cur_ind += 1
            if cur_ind >= len(cur_set[length]):
                random.shuffle(cur_set[length])
                cur_ind = 0
            counters[length] = cur_ind
            if offset is not None and offset + b < len(cur_set[length]):
                elem = cur_set[length][offset + b]
        else:
            elem = preset
        inp, target = elem[0], elem[1]
        assert len(inp) <= length, "Input len {inp}; Length {length}".format(inp=inp, length=length)

        padded_input, padded_target = add_padding(inp, target, max_length)
        inputs.append(padded_input)
        targets.append(padded_target)

    new_input = inputs
    new_target = targets
    return new_input, new_target


def add_padding(inp: list, target: list, max_length: int):
    if cnf.disperse_padding:
        inp, target = disperse_padding(inp, max_length, target)

    pad_len_input = max_length - len(inp)
    pad_len_output = max_length - len(target)
    pad_len_before = 0
    if cnf.use_front_padding:
        pad_len_before = np.random.randint(min(pad_len_input, pad_len_output) + 1)
    pad_before = np.zeros([pad_len_before])

    padded_input = np.concatenate([pad_before, inp, np.zeros([pad_len_input - pad_len_before])])
    padded_target = np.concatenate([pad_before, target, np.zeros([pad_len_output - pad_len_before])])

    return padded_input, padded_target


def disperse_padding(inp, max_length, target):
    assert len(inp) == len(target)
    desired_length = np.random.randint(len(inp), max_length + 1)
    cur_symbol = 0
    res_in = []
    res_out = []
    for i in range(desired_length):
        remaining_symbols = len(inp) - cur_symbol
        if np.random.randint(desired_length - i) >= remaining_symbols:
            res_in.append(0)
            res_out.append(0)
        else:
            res_in.append(inp[cur_symbol])
            res_out.append(target[cur_symbol])
            cur_symbol += 1
    remaining_symbols = len(inp) - cur_symbol
    assert remaining_symbols == 0
    assert len(res_in) == desired_length
    assert len(res_out) == desired_length

    return res_in, res_out


def print_out(s, newline=True):
    """Print a message out and log it to file."""
    if cnf.log_filename:
        try:
            with gfile.GFile(cnf.log_filename, mode="a") as f:
                f.write(s + ("\n" if newline else ""))
        except:
            sys.stdout.write("Error appending to %s\n" % cnf.log_filename)
    sys.stdout.write(s + ("\n" if newline else ""))
    sys.stdout.flush()


def accuracy(inpt, output, target, batch_size, nprint):
    """Calculate output accuracy given target."""
    assert nprint < batch_size + 1

    def task_print(inp, output, target):
        print_len = len(inp)
        print_out("    i: " + " ".join([str(i) for i in inp]))
        print_out("    o: " +
                  " ".join([str(output[L]) for L in range(print_len)]))
        print_out("    t: " +
                  " ".join([str(target[L]) for L in range(print_len)]))

    decoded_target = target
    decoded_output = output
    total = 0
    errors = 0
    seq = [0 for _ in range(batch_size)]
    for L in range(len(decoded_output[0])):
        for b in range(batch_size):
            if decoded_target[b][L] > 0 or decoded_output[b][L] > 0:
                total += 1
                if decoded_output[b][L] != decoded_target[b][L]:
                    seq[b] = 1
                    errors += 1
    e = 0  # Previous error index
    for _ in range(min(nprint, sum(seq))):
        while seq[e] == 0:
            e += 1
        task_print(inpt[e], decoded_output[e], decoded_target[e])
        e += 1
    return errors, total, sum(seq)


def print_bin_usage():
    test_cases = 0
    train_cases = 0
    test_cases_bins = 0
    train_cases_bins = 0

    test = test_set[cnf.task]
    train = train_set[cnf.task]

    for i in range(cnf.bin_max_len):
        train_cases += len(train[i])
        test_cases += len(test[i])

        if i in cnf.bins:
            test_cases_bins += len(test[i])
            train_cases_bins += len(train[i])

    print("\n------------------- BIN USAGE INFO -------------------")
    print("Train cases total:", train_cases, "In bins:", train_cases_bins)
    print("Test cases total:", test_cases, "; In bins:", test_cases_bins)
    print()
