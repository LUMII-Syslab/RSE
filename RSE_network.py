"""Implementation of the Shuffle-Exchange network parts of the RSE."""

import numpy as np
import tensorflow as tf

saturation_limit = 0.9
is_training = None
saturation_costs = []  # here we will collect all saturation costs
gate_mem = []
reset_mem = []
prev_mem_list = []
residual_list = []
candidate_mem = []
info_alpha = []
dropout_keep_prob = 1.0
add_noise = True


def gelu(x):
    return x * tf.sigmoid(1.702 * x)


def ror(x, n, p=1):
    """Bitwise rotation right p positions
    n is the bit length of the number
    """
    return (x >> p) + ((x & ((1 << p) - 1)) << (n - p))


def rol(x, n, p=1):
    """Bitwise rotation left p positions
    n is the bit length of the number
    """
    return ((x << p) & ((1 << n) - 1)) | (x >> (n - p))


def dropout(d, len):
    """Dropout dependent on sequence length"""
    if dropout_keep_prob < 1:
        prob = (1.0 - dropout_keep_prob) / len
        if is_training:
            d = tf.nn.dropout(d, rate=prob)
    return d


def add_noise_add(d, noise_scale):
    """Additive noise"""
    if is_training:
        d = d + tf.random_normal(tf.shape(d), stddev=noise_scale)
    return d


def inv_sigmoid(y):
    return np.log(y / (1 - y))


def layer_norm(cur, scope):
    """Normalize based on mean variance"""
    with tf.variable_scope(scope):
        cur -= tf.reduce_mean(cur, axis=1, keepdims=True)
        cur += add_bias_1(cur, "norm_bias")
        variance = tf.reduce_mean(tf.square(cur), [1], keepdims=True)
        cur = cur * tf.rsqrt(variance + 1e-10)
        return cur


def add_bias_1(cur, scope):
    with tf.variable_scope(scope):
        size = cur.get_shape().as_list()
        offset = tf.get_variable('offset', [1, 1, size[-1]], initializer=tf.zeros_initializer)
        return cur + offset


def conv_linear(input, kernel_width, nin, nout, bias_start, prefix, add_bias=True, init_scale=1.0, stride=1):
    """Convolutional linear map"""

    with tf.variable_scope(prefix):
        initializer = tf.variance_scaling_initializer(scale=init_scale, mode="fan_avg", distribution="uniform")
        if kernel_width == 1:
            inp_shape = input.get_shape().as_list()
            filter = tf.get_variable("CvK", [nin, nout], initializer=initializer)

            res = tf.matmul(tf.reshape(input, [inp_shape[0] * inp_shape[1], nin]), filter)
            res = tf.reshape(res, [inp_shape[0], inp_shape[1], nout])
        else:
            filter = tf.get_variable("CvK", [kernel_width, nin, nout], initializer=initializer)
            res = tf.nn.conv1d(input, filter, stride, "SAME")

        if add_bias:
            # nonzero initializer is used to prevent degenerancy issues with normalization and zero inputs
            bias_term = tf.get_variable("CvB", [nout], initializer=tf.random_uniform_initializer(
                bias_start, bias_start + 0.01))
            res = res + bias_term

        return res


def shuffle_layer(mem, do_ror=True):
    """Shuffles the elements according to bitwise left or right rotation on their indices"""
    length = mem.get_shape().as_list()[1]
    n_bits = (length - 1).bit_length()
    if do_ror:
        rev_indices = [ror(x, n_bits) for x in range(length)]
    else:
        rev_indices = [rol(x, n_bits) for x in range(length)]
    mem_shuffled = tf.gather(mem, rev_indices, axis=1)
    return mem_shuffled

def info_dropout(inputs, prefix="infodrop", initial_value=-6, desired_value = 0.0):
    """
    Information dropout
    Heavily modified from https://arxiv.org/abs/1611.01353
    """

    def continous_dropout(shape, noise_scale):
        """
        one sided multiplicative noise from beta-distribution
        a good range for noise_scale is (-10, 1). Small values give less noise
        noise_scale = 0 gives uniform distribution
        """
        noise_scale = tf.exp(-noise_scale)
        n = tf.random_uniform(shape, 0.0, 1.0)
        expected_value = noise_scale/(noise_scale + 1)
        noise = (1 - tf.pow(n, noise_scale))
        if is_training:
            return noise #noise value during training
        else:
             return expected_value #expected value at test time

    num_units = inputs.get_shape().as_list()[2]
    log_sigma_sq = conv_linear(inputs, 1, num_units, num_units, 0.0, prefix + "/infodrop") + initial_value
    KL_loss0 = -(1 + log_sigma_sq - desired_value - tf.exp(log_sigma_sq - desired_value))
    tf.summary.histogram(prefix + '/infohistogram', log_sigma_sq)
    info_alpha.append(log_sigma_sq)
    kl = tf.reduce_mean(KL_loss0)
    tf.add_to_collection('kl_terms', kl) # !!! NB these values should be added to the loss function with a small weight !!!
    log_sigma_sq = tf.minimum(log_sigma_sq, desired_value) #clip to thedesired minimum value
    e = continous_dropout(inputs.shape, log_sigma_sq)
    return inputs * e


def switch_layer(mem_shuffled, kernel_width, prefix):
    """Computation unit for every two adjacent elements"""
    length = mem_shuffled.get_shape().as_list()[1]
    num_units = mem_shuffled.get_shape().as_list()[2]
    batch_size = mem_shuffled.get_shape().as_list()[0]
    n_bits = (length - 1).bit_length()

    def conv_lin_reset_relu(input, suffix, in_units, out_units):
        middle_units = in_units * 4
        res = conv_linear(input, kernel_width, in_units * 2, middle_units, 0.0, prefix + "/cand1/" + suffix,
                          add_bias=False, init_scale=1.0)
        res = layer_norm(res, prefix + "/norm/" + suffix)
        res_middle = res
        res = gelu(res)
        res = conv_linear(res, kernel_width, middle_units, out_units * 2, 0.0, prefix + "/cand2/" + suffix,
                          init_scale=1.0)
        return res, res_middle

    mem_shuffled_x = mem_shuffled
    mem_all = mem_shuffled
    in_maps = num_units

    # calculate the new value
    mem_all = tf.reshape(mem_all, [batch_size, length // 2, in_maps * 2])
    candidate, reset = conv_lin_reset_relu(mem_all, "c", in_maps, num_units)
    reset_mem.append(tf.reshape(reset, [batch_size, length, -1])[:, :, :num_units])
    candidate_mem.append(tf.reshape(candidate, [batch_size, length, num_units]))

    residual_weight = 0.9
    candidate_weight = np.sqrt(1 - residual_weight ** 2) * 0.25
    lr_adjust = 2
    residual_scale = tf.sigmoid(tf.get_variable(prefix + "/residual", [num_units * 2],
                                                initializer=tf.constant_initializer(
                                                    inv_sigmoid(residual_weight) / lr_adjust)) * lr_adjust)
    residual_list.append(tf.reshape(tf.clip_by_value(residual_scale, 0.0, 1.0), [1, num_units * 2, 1]))

    mem_shuffled_x = tf.reshape(mem_shuffled_x, [batch_size, length // 2, in_maps * 2])
    candidate = residual_scale * mem_shuffled_x + candidate * candidate_weight
    candidate = tf.reshape(candidate, [batch_size, length, num_units])
    #candidate = dropout(candidate, n_bits)
    candidate = info_dropout(candidate, prefix=prefix)
    candidate = add_noise_add(candidate, 0.01)

    return candidate


def shuffle_exchange_network_heavy_sharing(cur, name, kernel_width=1, n_blocks=1, tied_inner_weights=True,
                                           tied_outer_weights=False):
    """Neural Benes Network with skip connections between blocks."""
    length = cur.get_shape().as_list()[1]
    n_bits = (length - 1).bit_length()
    all_mem = []

    with tf.variable_scope(name + "_recursive", reuse=tf.AUTO_REUSE):
        for k in range(n_blocks):
            outstack = []
            for i in range(n_bits - 1):
                outstack.append(cur)
                layer_name = "forward"
                if not tied_outer_weights: layer_name = str(k) + "_" + layer_name
                if not tied_inner_weights: layer_name += "_" + str(i)
                cur = switch_layer(cur, kernel_width, layer_name)
                all_mem.append(cur)
                cur = shuffle_layer(cur, do_ror=False)

            for i in range(n_bits - 1):
                outstack.append(cur)
                layer_name = "reverse"
                if not tied_outer_weights: layer_name = str(k) + "_" + layer_name
                if not tied_inner_weights: layer_name += "_" + str(n_bits - 1 - 1 - i)
                cur = switch_layer(cur, kernel_width, layer_name)
                all_mem.append(cur)
                cur = shuffle_layer(cur, do_ror=True)

        layer_name = "last"
        cur = switch_layer(cur, kernel_width, layer_name)
        all_mem.append(cur)

    return cur, all_mem


def shuffle_exchange_network(cur, name, kernel_width=1, n_blocks=1, tied_inner_weights=True, tied_outer_weights=False):
    """Neural Benes Network with residual connections between blocks."""
    length = cur.get_shape().as_list()[1]
    n_bits = (length - 1).bit_length()
    all_mem = []

    with tf.variable_scope(name + "/shuffle_exchange", reuse=tf.AUTO_REUSE):
        outstack = []
        stack = []

        def switch_and_shuffle(cur, do_ror, layer_name, block_index, layer_index):
            prev = stack[layer_index] if len(stack) > 0 else None
            if not tied_outer_weights or prev is None: layer_name = str(block_index) + "_" + layer_name
            if not tied_inner_weights: layer_name += "_" + str(layer_index)
            cur = switch_layer(cur, kernel_width, layer_name)
            cur = shuffle_layer(cur, do_ror=do_ror)
            outstack.append(cur)
            all_mem.append(cur)
            return cur

        for k in range(n_blocks):
            layer_ind = 0
            outstack = [cur]
            cur = switch_and_shuffle(cur, False, "first_layer", k, layer_ind)
            layer_ind += 1

            for i in range(n_bits - 2):
                cur = switch_and_shuffle(cur, False, "forward", k, layer_ind)
                layer_ind += 1

            cur = switch_and_shuffle(cur, True, "middle_layer", k, layer_ind)
            layer_ind += 1

            for i in range(n_bits - 2):
                cur = switch_and_shuffle(cur, True, "backward", k, layer_ind)
                layer_ind += 1

            stack = outstack

        cur = switch_layer(cur, kernel_width, "last_layer")
        all_mem.append(cur)

    return cur, all_mem
