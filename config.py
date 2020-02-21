"""
    Tensorflow configuration
"""
import numpy as np
import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

"""
    Model configuration
"""
input_type = tf.int64
dropout_keep_prob = 0.9
input_dropout_keep_prob = 1.0
output_dropout_keep_prob = 1.0
input_word_dropout_keep_prob = 1.0
label_smoothing = 0.01
embedding_size = None

""" 
    Test configuration
"""
max_test_length = 5000
test_data_size = 1024

"""
    Local storage (checkpoints, etc).
"""
use_two_gpus = False
gpu_instance = "1"
out_dir = "/host-dir/gpu" + gpu_instance
model_file = out_dir + "/varWeights.ckpt"
image_path = out_dir + "/images"

"""
    Logging
"""
log_filename = ""

"""
    Training config
"""
training_iters = 20000
display_step = 100
batch_size = 32
data_size = 10000
bins = [8, 16, 32, 64]
n_Benes_blocks = 2
load_prev = False

num_warmup_steps = 0

"""
    MusicNet configuration
"""
musicnet_data_dir = "/host-dir/musicnet"
musicnet_vocab_size = 128  # number of labels (notes)

# for data loading in musicnet.py:
musicnet_resample = False  # if true, resample partial train set at partial validation time
musicnet_mmap_load = False  # use mmap to load the full training dataset
musicnet_mmap_partial = False  # use mmap to sample a partial training dataset
musicnet_mmap_count = 500  # how many inputs to sample for partial training dataset

musicnet_test_step = 1000  # each x steps partial validation tests are launched
musicnet_full_test_step = 10000  # each x steps full validation test is launched
musicnet_n_test_batches = 30  # n of batches for partial validation AveragePrecisionScore

"""
    Lambada configuration
"""
lambada_data_dir = "/host-dir/lambada-dataset"
lambada_vocab_size = 999996

# Data preparation
use_front_padding = False  # randomly shift the starting position of the sequence in the bin
disperse_padding = False  # insert random blanks in the sequence

"""
    Embedding configuration
"""
use_pre_trained_embedding = False
base_folder = "/host-dir/embeddings/"
embedding_file = base_folder + "fast_word_embedding.vec"
emb_vector_file = base_folder + "emb_vectors.bin"
emb_word_dictionary = base_folder + "word_dict.bin"

"""
    Task configuration.
"""
all_tasks = {"sort", "kvsort", "id", "rev", "rev2", "incr", "add", "left",
             "right", "bmul", "mul", "dup",
             "badd", "qadd", "search", "qmul", "mulbcd", "shuffle", "div",
             "w_sort", "lambada",
             "dyck", "dyck_continue", "rol", "memory_indexing",
             "musicnet"}

language_tasks = {"lambada", "musicnet"}

# suggested settings for binary addition
# task = "badd"
# n_input = 13  # range of input digits
# n_output = 4  # range of output digits
# n_hidden = 48 * 4  # number of maps

# suggested settings for rotation
# task = "rol"
# n_input = 5  # range of input digits
# n_output = 5  # range of output digits
# n_hidden = 48 * 2  # number of maps

# suggested settings for memory indexing
# task = "memory_indexing"
# n_input = 129*3+1  # range of input digits
# n_output = 3  # range of output digits
# n_hidden = 48 * 4  # number of maps

# task = "shuffle"
# n_input = 12  # range of input digits
# n_output = 12  # range of output digits
# n_hidden = 48 * 2  # number of maps

# suggested settings for word lexicographic sorting
# task = "w_sort"
# n_input = 100
# n_output = 100
# n_hidden = 48 * 2

# settings for Dyck language parsing
# task = "dyck"
# n_input = 5  # Matching values are made as 1-2 3-4 5-6 ....
# n_output = 5
# n_hidden = 48 * 2

# settings for Dyck language last bracket task (input - Dyck word without last bracket, output - matching bracket)
# task = "dyck_continue"
# n_input = 5
# n_output = 5
# n_hidden = 48 * 2

# suggested settings for binary multiplication
# task = "bmul"
# n_input = 13  # range of input digits
# n_output = 4  # range of output digits
# n_hidden = 48 * 8  # number of maps

# task = "div"
# n_input = 13 # range of input digits
# n_output = 6 # range of output digits
# n_hidden = 48*8 # number of maps

# suggested settings for binary addition
# task = "qadd"
# n_input = 13 #range of input digits
# n_output = 5 #range of output digits
# n_hidden = 48*2 # number of maps
# dropout_keep_prob = 0.5

# suggested settings for base-4 multiplication
# task = "qmul"
# n_input = 13
# n_output = 5
# n_hidden = 48*4 # number of maps

# suggested settings for decimal multiplication with binary encoding
# task = "mulbcd"
# n_input = 13
# n_output = 5
# import data_utils as data_gen
# data_gen.bins = [9, 17, 25, 33, 41]  # we have to specify different bins here since for many lengths we have no examples to train
# n_hidden = 48 * 4  # number of maps

# suggested settings for sorting numbers in range 0 to 5
# task = "sort"
# n_input = 12
# n_output = 12
# n_hidden = 48*4 # number of maps
# n_Benes_blocks = 1

# task = "kvsort"
# n_input = 12
# n_output = 12
# n_hidden = 48 * 2  # number of maps

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

task = "musicnet"
input_type = tf.float32
n_input = musicnet_vocab_size
n_output = musicnet_vocab_size
n_hidden = 48 * 4
input_word_dropout_keep_prob = 1.0
label_smoothing = 0.01
embedding_size = 1
max_test_length = 10000
test_data_size = 10000
musicnet_window_size = 4096  # 128 .. 8192
training_iters = 400000
batch_size = 32
n_Benes_blocks = 2
bins = [musicnet_window_size]

initial_learning_rate = 0.00125 * np.sqrt(96 / n_hidden)
min_learning_rate = initial_learning_rate
if load_prev:
    initial_learning_rate = min_learning_rate

forward_max = bins[-1]
bin_max_len = max(max_test_length, forward_max)
