import numpy as np
import tensorflow as tf

"""Tensorflow configuration"""
tf_config = tf.ConfigProto()
tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

"""Local storage (checkpoints, etc)."""
use_two_gpus = False
gpu_instance = "0"
out_dir = "/host-dir/gpu" + gpu_instance
model_file = out_dir + "/varWeights.ckpt"
image_path = out_dir + "/images"
log_filename = ""

"""Model configuration"""
input_type = tf.int64
dropout_keep_prob = 0.9
input_dropout_keep_prob = 1.0
output_dropout_keep_prob = 1.0
input_word_dropout_keep_prob = 1.0
label_smoothing = 0.01
embedding_size = None

"""Training configuration"""
training_iters = 20000
display_step = 100
batch_size = 32
data_size = 10000
bins = [8, 16, 32, 64]
n_Benes_blocks = 2
load_prev = False  # load a saved model
num_warmup_steps = 0

"""Test configuration"""
max_test_length = 5000
test_data_size = 1024

"""MusicNet configuration"""
musicnet_window_size = 8192  # power of two within [256, 8192]
musicnet_do_fourier_transform = True  # use dataset pre-processed with Fourier transform
musicnet_fourier_multiplier = 1  # use dataset with first values from x times larger Fourier window (1 = whole window)
musicnet_file_window_size = 8192  # select .npy dataset files with this non-cropped window size
musicnet_visualise = False  # disables validation data shuffling for better visualisation
musicnet_subset = True  # loads a subset of train_set to save RAM. Frequently loads a different subset with np memmap
musicnet_mmap_count = 10000  # how many inputs to load for the train_set subset
musicnet_test_step = 1000  # each x steps partial validation tests are launched (validates on a subset to save time)
musicnet_n_test_batches = 100  # n of batches for partial validation
musicnet_full_test_step = 100000  # each x steps full validation test is launched
musicnet_vocab_size = 128  # number of labels (notes)

"""Lambada configuration"""
lambada_data_dir = "/host-dir/lambada-dataset"
lambada_vocab_size = 999996

"""Data preparation"""
use_front_padding = False  # randomly shift the starting position of the sequence in the bin
disperse_padding = False  # insert random blanks in the sequence

"""Embedding configuration"""
use_pre_trained_embedding = False
base_folder = "/host-dir/embeddings/"
embedding_file = base_folder + "fast_word_embedding.vec"
emb_vector_file = base_folder + "emb_vectors.bin"
emb_word_dictionary = base_folder + "word_dict.bin"

"""Task configuration"""
all_tasks = {"sort", "kvsort", "id", "rev", "rev2", "incr", "add", "left", "right", "bmul", "mul", "dup", "badd",
             "qadd", "search", "qmul", "mulbcd", "shuffle", "div", "w_sort", "lambada", "dyck", "dyck_continue", "rol",
             "memory_indexing", "musicnet"}
language_tasks = {"lambada", "musicnet"}

"""Recommended settings for binary addition"""
# task = "badd"
# n_input = 13  # range of input digits
# n_output = 4  # range of output digits
# n_hidden = 48 * 4  # number of maps

"""Recommended settings for rotation"""
# task = "rol"
# n_input = 5  # range of input digits
# n_output = 5  # range of output digits
# n_hidden = 48 * 2  # number of maps

"""Recommended settings for memory indexing"""
# task = "memory_indexing"
# n_input = 129*3+1  # range of input digits
# n_output = 3  # range of output digits
# n_hidden = 48 * 4  # number of maps

"""Recommended settings for shuffle"""
# task = "shuffle"
# n_input = 12  # range of input digits
# n_output = 12  # range of output digits
# n_hidden = 48 * 2  # number of maps

"""Recommended settings for word lexicographic sorting"""
# task = "w_sort"
# n_input = 100
# n_output = 100
# n_hidden = 48 * 2

"""Recommended settings for Dyck language parsing"""
# task = "dyck"
# n_input = 5  # Matching values are made as 1-2 3-4 5-6 ....
# n_output = 5
# n_hidden = 48 * 2

"""Recommended settings for Dyck language last bracket task (input - Dyck word without last bracket, output - matching bracket)"""
# task = "dyck_continue"
# n_input = 5
# n_output = 5
# n_hidden = 48 * 2

"""Recommended settings for binary multiplication"""
# task = "bmul"
# n_input = 13  # range of input digits
# n_output = 4  # range of output digits
# n_hidden = 48 * 8  # number of maps

"""Recommended settings for division"""
# task = "div"
# n_input = 13 # range of input digits
# n_output = 6 # range of output digits
# n_hidden = 48*8 # number of maps

"""Recommended settings for base-4 addition"""
# task = "qadd"
# n_input = 13 #range of input digits
# n_output = 5 #range of output digits
# n_hidden = 48*2 # number of maps
# dropout_keep_prob = 0.5

"""Recommended settings for base-4 multiplication"""
# task = "qmul"
# n_input = 13
# n_output = 5
# n_hidden = 48*4 # number of maps

"""Recommended settings for decimal multiplication with binary encoding"""
# task = "mulbcd"
# n_input = 13
# n_output = 5
# import data_utils as data_gen
# data_gen.bins = [9, 17, 25, 33, 41]  # we have to specify different bins here since for many lengths we have no examples to train
# n_hidden = 48 * 4  # number of maps

"""Recommended settings for sorting numbers in range 0 to 5"""
# task = "sort"
# n_input = 12
# n_output = 12
# n_hidden = 48*4 # number of maps
# n_Benes_blocks = 1

"""Recommended settings for kvsort"""
# task = "kvsort"
# n_input = 12
# n_output = 12
# n_hidden = 48 * 2  # number of maps

"""Recommended settings for lambada"""
# task = "lambada"
# n_input = lambada_vocab_size
# n_output = 3
# n_hidden = 48*8
# #input_dropout_keep_prob = 1.0
# input_word_dropout_keep_prob = 0.95
# use_front_padding = True
# use_pre_trained_embedding = True
# disperse_padding = False
# label_smoothing = 0.1
# batch_size = 64
# bins = [256]

"""Recommended settings for MusicNet"""
task = "musicnet"
n_Benes_blocks = 2  # depth of the model
n_hidden = 48 * 4  # first layer size of RSU (2m)
batch_size = 16
training_iters = 800000 + 1
input_word_dropout_keep_prob = 1.0
label_smoothing = 0.01
embedding_size = 1
max_test_length = 10000
test_data_size = 10000
input_type = tf.float32
n_input = musicnet_vocab_size
n_output = musicnet_vocab_size
bins = [musicnet_window_size]
musicnet_resample_step = musicnet_mmap_count // (batch_size+1)  # each x steps resamples if musicnet_resample == True


initial_learning_rate = 0.00125 * np.sqrt(96 / n_hidden)
min_learning_rate = initial_learning_rate
if load_prev:
    initial_learning_rate = min_learning_rate

forward_max = bins[-1]
bin_max_len = max(max_test_length, forward_max)
