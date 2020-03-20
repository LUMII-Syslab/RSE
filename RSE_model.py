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

"""Tensorflow implementation of the Residual Shuffle-Exchange model"""
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

import RSE_network
import config as cnf
from RAdam import RAdamOptimizer


class ModelSpecific:
    """
    Task specific model structure
    """

    def cost(self, prediction) -> tuple:
        """
        :rtype: tuple (cost, per_item_cost)
        """
        pass

    def accuracy(self, prediction):
        """
        :return: Accuracy as float tensor (single)
        """
        pass

    def result(self, prediction):
        pass

    @staticmethod
    def calculate_loss_with_smoothing(label, logits, output_classes, label_smoothing=0.0):
        # returns per example loss
        confidence = 1 - label_smoothing
        low_confidence = label_smoothing / (output_classes - 1)

        labels_one_hot = tf.one_hot(label, output_classes, on_value=confidence, off_value=low_confidence)

        # reduce the weight of the padding symbol
        mask_out = tf.cast(tf.not_equal(label, 0), tf.float32)
        weights = mask_out * 0.99 + 0.01
        weights /= tf.reduce_mean(weights, -1, keepdims=True)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)

        # the minimum cross_entropy achievable with label smoothing
        min_loss = -(confidence * np.log(confidence) + (output_classes - 1) *
                     low_confidence * np.log(low_confidence + 1e-20))

        return tf.reduce_mean((loss - min_loss) * weights, -1)


class LambadaModel(ModelSpecific):

    def __init__(self, target, n_classes, label_smoothing) -> None:
        self.__target = target
        self.__n_classes = n_classes
        self.__y_one_hot = tf.one_hot(self.__target, self.__n_classes, dtype=tf.float32)
        self.__label_smoothing = label_smoothing

    def cost(self, prediction):
        labels = self.__y_one_hot[:, :, 2] / tf.reduce_sum(self.__y_one_hot[:, :, 2], axis=1, keepdims=True)
        smooth_positives = 1.0 - self.__label_smoothing
        smooth_negatives = self.__label_smoothing / labels.get_shape().as_list()[1]
        onehot_labels = labels * smooth_positives + smooth_negatives
        cost_vector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction[:, :, 2], labels=onehot_labels)
        return tf.reduce_mean(cost_vector), cost_vector

    def accuracy(self, prediction):
        positions = tf.argmax(prediction[:, :, 2], axis=1)
        selected = self.__y_one_hot[:, :, 2]

        batch_index = tf.expand_dims(tf.range(positions.shape[0], dtype=tf.int64), axis=1)
        positions = tf.expand_dims(positions, axis=1)
        indices = tf.concat((batch_index, positions), axis=1)
        accuracy1 = tf.gather_nd(selected, indices)
        return tf.reduce_mean(accuracy1)

    def result(self, prediction):
        return tf.argmax(prediction[:, :, 2], axis=1)


class DefaultModel(ModelSpecific):
    def __init__(self, target, n_classes, label_smoothing) -> None:
        self.__target = target
        self.__n_classes = n_classes
        self.__label_smoothing = label_smoothing

    def cost(self, prediction):
        loss = self.calculate_loss_with_smoothing(self.__target, prediction, self.__n_classes, self.__label_smoothing)
        return tf.reduce_mean(loss), loss

    @staticmethod
    def get_accuracy(prediction, y_in):
        result = tf.argmax(prediction, 2)
        correct_symbols = tf.equal(result, y_in)
        mask_y_in = tf.cast(tf.not_equal(y_in, 0), tf.float32)
        mask_out = tf.cast(tf.not_equal(result, 0), tf.float32)
        mask_2 = tf.maximum(mask_y_in, mask_out)
        correct_symbols = tf.cast(correct_symbols, tf.float32)
        correct_symbols *= mask_2
        return tf.reduce_sum(correct_symbols, 1) / tf.reduce_sum(mask_2, 1)

    def bpc(self, prediction, name="bpc"):
        """ bits per character. Uses the first symbol only"""
        with tf.variable_scope(name):
            prediction = tf.stop_gradient(prediction[:, 0, :])  # disable learning from bpc
            # scale to undo label smoothing
            scale = tf.get_variable('scale', (), initializer=tf.ones_initializer)
            prediction = prediction * scale  # +offset
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.__target[:, 0], logits=prediction)
        return tf.reduce_mean(loss) / np.log(2)

    def accuracy(self, prediction):
        accuracy1 = self.get_accuracy(prediction, self.__target)
        return tf.reduce_mean(accuracy1)

    def result(self, prediction):
        return tf.argmax(prediction, axis=2)


class MusicNetModel(ModelSpecific):
    def __init__(self, target, n_classes, label_smoothing) -> None:
        self.__target = target
        self.__n_classes = n_classes
        self.__label_smoothing = label_smoothing

    def transformed_prediction(self, prediction):
        pred_1 = prediction[:, 0, :]  # gets 'note is played' predictions on 128 notes without padding
        pred_1 -= 4  # correct for class imbalance
        return pred_1

    def cost(self, prediction):
        pred_1 = self.transformed_prediction(prediction)
        labels_1 = tf.cast(self.__target[:, :128] - 1, tf.float32)  # gets labels on 128 notes without padding
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_1, logits=pred_1,
                                               label_smoothing=self.__label_smoothing)

        # add some small loss for unused entries
        pred_2 = prediction[:, 1:, :] - 4
        loss2 = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(pred_2), logits=pred_2,
                                                label_smoothing=(self.__label_smoothing + 0.1) / 2)
        total_loss = tf.reduce_mean(loss) + tf.reduce_mean(loss2) * 0.01

        return total_loss, loss

    def calibrated_result(self, prediction):
        with tf.variable_scope("corrected_result"):
            prediction = tf.stop_gradient(self.transformed_prediction(prediction))
            # scale to undo label smoothing
            offset = tf.get_variable('offset', (prediction.shape[-1]), initializer=tf.zeros_initializer)
            scale = tf.get_variable('scale', (prediction.shape[-1]), initializer=tf.ones_initializer)
            prediction = prediction * scale + offset
            labels_1 = tf.cast(self.__target[:, :128] - 1, tf.float32)  # gets labels on 128 notes without padding
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_1, logits=prediction)
            corrected_result = tf.sigmoid(prediction)

        return corrected_result, loss

    def accuracy(self, prediction):
        pred_1 = tf.sigmoid(self.transformed_prediction(prediction))
        labels_1 = tf.cast(self.__target[:, :128] - 1, tf.float32)  # gets labels on 128 notes without padding
        accuracy = tf.cast(tf.equal(tf.round(pred_1), labels_1), tf.float32)
        return tf.reduce_mean(accuracy)

    def result(self, prediction):
        return tf.sigmoid(self.transformed_prediction(prediction))


class MusicNetLateralModel(ModelSpecific):
    def __init__(self, target, n_classes, label_smoothing) -> None:
        self.__target = target
        self.__n_classes = n_classes
        self.__label_smoothing = label_smoothing
        self.conv_downscale = 4  # conv_pool_block2 downscales 4 times
        self.stride_labels = 128  # segment is labeled at positions with this stride
        self.n_frames = cnf.musicnet_window_size // self.stride_labels - 1  # -1 to exclude edges

    def transformed_prediction(self, prediction):
        transformed_pred = []
        for i in range(self.n_frames):
            transformed_pred += [prediction[:, i*self.stride_labels//self.conv_downscale, :]-4]  # -4 to correct for class imbalance
        return transformed_pred

    def unflatten_labels(self):
        unflattened_labels = []
        for i in range(self.n_frames):
            unflattened_labels += [self.__target[:, i*self.stride_labels:i*self.stride_labels+128]-1]  # -1 to get 0/1 labels
        return unflattened_labels

    def cost(self, prediction):
        transformed_pred = self.transformed_prediction(prediction)
        unflattened_labels = self.unflatten_labels()
        loss_lateral = 0
        for i in range(self.n_frames):
            loss_lateral += tf.losses.sigmoid_cross_entropy(
                multi_class_labels=unflattened_labels[i], logits=transformed_pred[i], label_smoothing=self.__label_smoothing)
        loss_mid = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=unflattened_labels[0], logits=transformed_pred[0], label_smoothing=self.__label_smoothing)

        # add some small loss for non-mid and unused entries
        pred_others = prediction[:, 1:, :] - 4
        loss_others = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(pred_others), logits=pred_others,
                                                label_smoothing=(self.__label_smoothing + 0.1) / 2)

        lateral_coef = 2 * 1/self.n_frames
        total_loss = tf.reduce_mean(loss_mid) + tf.reduce_mean(loss_lateral)*lateral_coef + tf.reduce_mean(loss_others) * 0.01

        return total_loss, loss_mid

    def calibrated_result(self, prediction):
        with tf.variable_scope("corrected_result"):
            prediction = tf.stop_gradient(self.transformed_prediction(prediction)[0])
            # scale to undo label smoothing
            offset = tf.get_variable('offset', (prediction.shape[-1]), initializer=tf.zeros_initializer)
            scale = tf.get_variable('scale', (prediction.shape[-1]), initializer=tf.ones_initializer)
            prediction = prediction * scale + offset
            labels = tf.cast(self.__target[:, :128] - 1, tf.float32)  # gets labels on 128 notes without padding
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=prediction)
            corrected_result = tf.sigmoid(prediction)

        return corrected_result, loss

    def accuracy(self, prediction):
        pred_1 = tf.sigmoid(self.transformed_prediction(prediction)[0])
        labels_1 = tf.cast(self.__target[:, :128] - 1, tf.float32)  # gets labels on 128 notes without padding
        accuracy = tf.cast(tf.equal(tf.round(pred_1), labels_1), tf.float32)
        return tf.reduce_mean(accuracy)

    def result(self, prediction):
        return tf.sigmoid(self.transformed_prediction(prediction)[0])


class MusicNetLateralOrderedModel(ModelSpecific):
    def __init__(self, target, n_classes, label_smoothing) -> None:
        self.__target = target
        self.__n_classes = n_classes
        self.__label_smoothing = label_smoothing
        self.conv_downscale = 4  # conv_pool_block2 downscales 4 times
        self.stride_labels = 128  # segment is labeled at positions with this stride
        self.n_frames = cnf.musicnet_window_size // self.stride_labels - 1  # -1 to exclude edges

    def transformed_prediction(self, prediction):
        transformed_pred = []
        for i in range(self.n_frames):
            transformed_pred += [prediction[:, i*self.stride_labels//self.conv_downscale, :]-4]  # -4 to correct for class imbalance
        # for ordered lateral labels:
        # transformed_pred[0], transformed_pred[1 + self.n_frames // 2] = transformed_pred[1 + self.n_frames // 2], transformed_pred[0]
        return transformed_pred

    def unflatten_labels(self):
        unflattened_labels = []
        for i in range(self.n_frames):
            unflattened_labels += [self.__target[:, i*self.stride_labels:i*self.stride_labels+128]-1]  # -1 to get 0/1 labels
        # for ordered lateral labels:
        # unflattened_labels[0], unflattened_labels[1 + self.n_frames // 2] = unflattened_labels[1 + self.n_frames // 2], unflattened_labels[0]
        return unflattened_labels

    def cost(self, prediction):
        transformed_pred = self.transformed_prediction(prediction)
        unflattened_labels = self.unflatten_labels()
        loss_lateral = 0
        for i in range(self.n_frames):
            loss_lateral += tf.losses.sigmoid_cross_entropy(
                multi_class_labels=unflattened_labels[i], logits=transformed_pred[i], label_smoothing=self.__label_smoothing)
        loss_mid = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=unflattened_labels[self.n_frames//2], logits=transformed_pred[self.n_frames//2], label_smoothing=self.__label_smoothing)

        # add some small loss for all entries to reduce the unused ones:
        pred_others = prediction[:, 0:, :] - 4
        loss_others = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(pred_others), logits=pred_others,
                                                label_smoothing=(self.__label_smoothing + 0.1) / 2)

        lateral_coef = 2 * 1/self.n_frames
        total_loss = tf.reduce_mean(loss_mid) + tf.reduce_mean(loss_lateral)*lateral_coef + tf.reduce_mean(loss_others) * 0.01

        return total_loss, loss_mid

    def calibrated_result(self, prediction):
        # calibrating for the mid label
        with tf.variable_scope("corrected_result"):
            prediction = tf.stop_gradient(self.transformed_prediction(prediction)[self.n_frames//2])
            # scale to undo label smoothing
            offset = tf.get_variable('offset', (prediction.shape[-1]), initializer=tf.zeros_initializer)
            scale = tf.get_variable('scale', (prediction.shape[-1]), initializer=tf.ones_initializer)
            prediction = prediction * scale + offset
            unflattened_labels = self.unflatten_labels()[self.n_frames//2]
            labels = tf.cast(unflattened_labels, tf.float32)  # gets labels on 128 notes without padding
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=prediction)
            corrected_result = tf.sigmoid(prediction)
        return corrected_result, loss

    def accuracy(self, prediction):
        pred_1 = tf.sigmoid(self.transformed_prediction(prediction)[self.n_frames//2])
        labels_1 = tf.cast(self.unflatten_labels()[self.n_frames//2], tf.float32)  # gets labels on 128 notes without padding
        accuracy = tf.cast(tf.equal(tf.round(pred_1), labels_1), tf.float32)
        return tf.reduce_mean(accuracy)

    def result(self, prediction):
        # return predictions for mid
        return tf.sigmoid(self.transformed_prediction(prediction)[self.n_frames//2])


class DNGPU:
    def __init__(self, num_units, bins, n_input, count_list, n_classes, dropout_keep_prob,
                 create_translation_model=False, use_two_gpus=False):
        self.translation_model = create_translation_model
        self.use_two_gpus = use_two_gpus
        self.n_classes = n_classes
        self.n_input = n_input
        self.num_units = num_units
        self.embedding_size = self.num_units if cnf.embedding_size is None else cnf.embedding_size
        self.bins = bins
        self.count_list = count_list
        self.accuracy = None
        self.base_cost = None
        self.sat_loss = None
        self.optimizer = None
        self.cost_list = None
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.Variable(cnf.initial_learning_rate, trainable=False)
        self.beta2_rate = tf.maximum(0.0005,
                                     tf.train.exponential_decay(0.01, self.global_step, 2000, 0.5, staircase=False))
        self.bin_losses = []
        RSE_network.dropout_keep_prob = dropout_keep_prob
        self.allMem = None
        self.tmpfloat = tf.placeholder("float")
        self.saturation_weight = tf.Variable(1e-3, trainable=False)
        self.assign_saturation_weight_op = self.saturation_weight.assign(self.tmpfloat)
        self.x_input = []
        self.y_input = []
        self.test_x = None
        self.test_y = None
        self.lr_decay_op = self.learning_rate.assign(tf.maximum(cnf.min_learning_rate, self.learning_rate * 0.7))
        self.n_middle = 48
        self.variable_summaries = None
        RSE_network.is_training = True

        if cnf.use_pre_trained_embedding:
            with open(cnf.emb_vector_file, "rb") as emb_file:
                emb = pickle.load(emb_file)  # Load binary numpy array with embeddings

            with tf.device('/cpu:0'):
                self.embedding_initializer = tf.constant_initializer(emb, verify_shape=True)
                self.embedding_shape = emb.shape

    def add_discrete_noise_unk(self, x, replacement_probability):
        """Randomply replaces some elements of the sequence with "unk"=1 symbol"""
        x_unk = tf.constant(1, dtype=tf.int64)
        n_scale = tf.floor(tf.random_uniform(tf.shape(x)) + replacement_probability)
        n_scale_masked = tf.cast(n_scale, tf.int64)
        x_in_indices_rand = x * (1 - n_scale_masked) + x_unk * n_scale_masked
        return x_in_indices_rand

    def conv_pool_block(self, cur, kernel_width=4, name="pool1"):
        with tf.variable_scope(name):
            n_maps2 = self.num_units * 2
            cur = RSE_network.conv_linear(cur, kernel_width, self.embedding_size, n_maps2, 0.0, "conv1", add_bias=False,
                                          stride=2)
            cur = RSE_network.layer_norm(cur, "norm1")
            cur = RSE_network.gelu(cur)
            cur = RSE_network.conv_linear(cur, 1, n_maps2, self.num_units, 0.0, "conv2") * 0.25
        return cur

    def conv_pool_block2(self, cur, kernel_width=4, name="pool1"):
        with tf.variable_scope(name):
            n_maps1 = self.num_units // 2
            n_maps2 = self.num_units * 2

            cur = RSE_network.conv_linear(cur, kernel_width, self.embedding_size, n_maps1, 0.0, "conv1", add_bias=False,
                                          stride=2)
            cur = RSE_network.layer_norm(cur, "norm1")
            cur = RSE_network.gelu(cur)
            cur = RSE_network.conv_linear(cur, kernel_width, n_maps1, n_maps2, 0.0, "conv2", add_bias=False, stride=2)
            cur = RSE_network.layer_norm(cur, "norm2")
            cur = RSE_network.gelu(cur)
            cur = RSE_network.conv_linear(cur, 1, n_maps2, self.num_units, 0.0, "conv3") * 0.25
        return cur

    def conv_pool_block3(self, cur, kernel_width=4, name="pool1"):
        with tf.variable_scope(name):
            n_maps1 = self.num_units // 2
            n_maps2 = self.num_units
            n_maps3 = self.num_units * 2

            cur = RSE_network.conv_linear(cur, kernel_width, self.embedding_size, n_maps1, 0.0, "conv1", add_bias=False,
                                          stride=2)
            cur = RSE_network.layer_norm(cur, "norm1")
            cur = RSE_network.gelu(cur)
            cur = RSE_network.conv_linear(cur, kernel_width, n_maps1, n_maps2, 0.0, "conv2", add_bias=False, stride=2)
            cur = RSE_network.layer_norm(cur, "norm2")
            cur = RSE_network.gelu(cur)
            cur = RSE_network.conv_linear(cur, kernel_width, n_maps2, n_maps3, 0.0, "conv3", add_bias=False, stride=2)
            cur = RSE_network.layer_norm(cur, "norm3")
            cur = RSE_network.gelu(cur)

            cur = RSE_network.conv_linear(cur, 1, n_maps3, self.num_units, 0.0, "conv4") * 0.25
        return cur

    def create_loss(self, x_in_indices, y_in, length):
        """perform loss calculation for one bin """

        batch_size = self.count_list[0]
        if cnf.use_pre_trained_embedding:
            cur = self.pre_trained_embedding(x_in_indices)
        else:
            if cnf.task == "musicnet":
                cur = tf.expand_dims(x_in_indices, axis=-1)
                # cur = DCGRU.conv_linear(cur, 1, 1, self.num_units, 0.0, "output_conv")
                if cnf.input_word_dropout_keep_prob < 1 and RSE_network.is_training:
                    cur = tf.nn.dropout(cur, cnf.input_word_dropout_keep_prob, noise_shape=[batch_size, length, 1])
                cur = RSE_network.add_noise_add(cur, 0.001)  # to help layernorm with zero inputs
                cur = self.conv_pool_block2(cur, name='pool1')
            else:
                cur = self.embedding(x_in_indices)

        if cnf.input_word_dropout_keep_prob < 1 and RSE_network.is_training and cnf.task != "musicnet":
            cur = tf.nn.dropout(cur, cnf.input_word_dropout_keep_prob, noise_shape=[batch_size, length, 1])
        if cnf.input_dropout_keep_prob < 1 and RSE_network.is_training:
            cur = tf.nn.dropout(cur, cnf.input_dropout_keep_prob)

        cur, allMem = RSE_network.shuffle_exchange_network_heavy_sharing(cur, "steps", n_blocks=cnf.n_Benes_blocks)

        print(length, len(allMem))

        all_mem_tensor = tf.stack(allMem)
        if RSE_network.is_training:
            cur = tf.nn.dropout(cur, cnf.output_dropout_keep_prob)
        prediction = RSE_network.conv_linear(cur, 1, self.num_units, self.n_classes, 0.0, "output")

        if cnf.task == "lambada":
            model = LambadaModel(y_in, self.n_classes, cnf.label_smoothing)
        elif cnf.task == "musicnet":
            # model = MusicNetModel(y_in, self.n_classes, cnf.label_smoothing)
            model = MusicNetLateralOrderedModel(y_in, self.n_classes, cnf.label_smoothing)
        else:
            model = DefaultModel(y_in, self.n_classes, cnf.label_smoothing)

        cost, per_item_cost = model.cost(prediction)
        result = model.result(prediction)
        accuracy = model.accuracy(prediction)
        bpc = tf.constant(0.0)
        if cnf.task == "musicnet":
            result, corrected_loss = model.calibrated_result(prediction)
            cost += corrected_loss * 0.1

        return cost, accuracy, all_mem_tensor, prediction, per_item_cost, result, bpc

    def embedding(self, indices):
        emb_weights = tf.get_variable("embedding", [self.n_input, self.embedding_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.25))
        cur = tf.nn.embedding_lookup(emb_weights, indices)
        return cur

    def pre_trained_embedding(self, indices):
        emb_weights = tf.get_variable("embedding", self.embedding_shape, tf.float32,
                                      initializer=self.embedding_initializer, trainable=False)
        cur = tf.nn.embedding_lookup(emb_weights, indices)
        cur = RSE_network.conv_linear(cur, 1, self.embedding_shape[1], self.num_units, 0.0, "embedding_linear")
        return cur

    def create_test_graph(self, test_length):
        RSE_network.is_training = False
        """Creates graph for accuracy evaluation"""
        with vs.variable_scope("var_lengths"):
            item_count = self.count_list[0]
            self.test_x = tf.placeholder(cnf.input_type, [item_count, test_length])
            self.test_y = tf.placeholder("int64", [item_count, test_length])
            _, self.test_accuracy, self.allMem, _, _, self.result, bpc = self.create_loss(self.test_x, self.test_y,
                                                                                          test_length)
            test_summaries = [tf.summary.scalar("base/test_error", 1 - self.test_accuracy)]
            if cnf.task == "musicnet":
                pred_flat = tf.reshape(self.result, [-1])
                labels_flat = tf.reshape(self.test_y[:, :128] - 1, [-1])  # gets 0/1 labels on 128 notes without padding
                pred_flat = tf.clip_by_value(pred_flat, 0.0, 1.0)
                roc, update_op = tf.metrics.auc(
                    labels=labels_flat, predictions=pred_flat, curve='PR', summation_method='careful_interpolation')
                with tf.control_dependencies([update_op]):
                    test_summaries.append(tf.summary.scalar("PR", roc))

            self.test_summary = tf.summary.merge(test_summaries)

    def create_graph(self):
        RSE_network.is_training = True
        """Creates graph for training"""
        self.base_cost = 0.0
        self.accuracy = 0
        num_sizes = len(self.bins)
        self.cost_list = []
        sum_weight = 0
        self.bin_losses = []
        saturation_loss = []
        total_mean_loss = 0

        # Create all bins and calculate losses for them

        with vs.variable_scope("var_lengths"):
            for seqLength, itemCount, ind in zip(self.bins, self.count_list, range(num_sizes)):
                x_in = tf.placeholder(cnf.input_type, [itemCount, seqLength])
                y_in = tf.placeholder("int64", [itemCount, seqLength])
                self.x_input.append(x_in)
                self.y_input.append(y_in)
                RSE_network.saturation_costs = []
                RSE_network.gate_mem = []
                RSE_network.reset_mem = []
                RSE_network.candidate_mem = []
                RSE_network.prev_mem_list = []
                RSE_network.residual_list = []
                RSE_network.info_alpha = []

                if self.use_two_gpus:
                    device = "/device:GPU:" + ("0" if seqLength >= self.bins[-1] else "1")
                    with tf.device(device):
                        c, a, mem1, logits, per_item_cost, _, _ = self.create_loss(x_in, y_in, seqLength)
                else:
                    c, a, mem1, logits, per_item_cost, _, _ = self.create_loss(x_in, y_in, seqLength)

                weight = 1.0

                sat_cost = tf.add_n(RSE_network.saturation_costs) / (
                        seqLength * len(RSE_network.saturation_costs) * itemCount) if len(
                    RSE_network.saturation_costs) > 0 else 0
                saturation_loss.append(sat_cost * weight)
                self.bin_losses.append(per_item_cost)
                self.base_cost += c * weight
                sum_weight += weight
                self.accuracy += a
                self.cost_list.append(c)

                mean_loss = tf.reduce_mean(tf.square(mem1))
                total_mean_loss += mean_loss

                tf.get_variable_scope().reuse_variables()

        # calculate the total loss
        self.base_cost /= sum_weight
        self.accuracy /= num_sizes
        total_mean_loss /= num_sizes
        tf.summary.scalar("base/loss", self.base_cost)
        tf.summary.scalar("base/error", 1 - self.accuracy)
        tf.summary.scalar("base/error_longest", 1 - a)
        tf.summary.histogram("logits", logits)

        if cnf.task is not "musicnet":
            if RSE_network.gate_mem:
                gate_img = tf.stack(RSE_network.gate_mem)
                gate_img = gate_img[:, 0:1, :, :]
                gate_img = tf.cast(gate_img * 255, dtype=tf.uint8)
                tf.summary.image("gate", tf.transpose(gate_img, [3, 0, 2, 1]), max_outputs=16)
            if RSE_network.reset_mem:
                reset_img = tf.stack(RSE_network.reset_mem)
                reset_img = tf.clip_by_value(reset_img, -2, 2)
                tf.summary.histogram("reset", reset_img)
                reset_img = reset_img[:, 0:1, :, :]
                tf.summary.image("reset", tf.transpose(reset_img, [3, 0, 2, 1]), max_outputs=16)
            if RSE_network.prev_mem_list:
                prev_img = tf.stack(RSE_network.prev_mem_list)
                prev_img = prev_img[:, 0:1, :, :]
                prev_img = tf.cast(prev_img * 255, dtype=tf.uint8)
                tf.summary.image("prev_mem", tf.transpose(prev_img, [3, 0, 2, 1]), max_outputs=16)
            if RSE_network.residual_list:
                prev_img = tf.stack(RSE_network.residual_list)
                prev_img = prev_img[:, 0:1, :, :]
                prev_img = tf.cast(prev_img * 255, dtype=tf.uint8)
                tf.summary.image("residual_mem", tf.transpose(prev_img, [3, 0, 2, 1]), max_outputs=16)
            if RSE_network.info_alpha:
                prev_img = tf.stack(RSE_network.info_alpha)
                prev_img = prev_img[:, 0:1, :, :]
                tf.summary.image("info_alpha", tf.transpose(prev_img, [3, 0, 2, 1]), max_outputs=16)

            candidate_img = tf.stack(RSE_network.candidate_mem)
            candidate_img = candidate_img[:, 0:1, :, :]
            candidate_img = tf.cast((candidate_img + 1.0) * 127.5, dtype=tf.uint8)
            tf.summary.image("candidate", tf.transpose(candidate_img, [3, 0, 2, 1]), max_outputs=16)

            mem1 = mem1[:, 0:1, :, :]
            tf.summary.image("mem", tf.transpose(mem1, [3, 0, 2, 1]), max_outputs=16)

        saturation = tf.reduce_sum(tf.stack(saturation_loss)) / sum_weight
        tf.summary.scalar("base/activation_mean", tf.sqrt(total_mean_loss))

        self.sat_loss = saturation * self.saturation_weight
        cost = self.base_cost + self.sat_loss

        kl_terms = tf.get_collection('kl_terms')
        kl_sum = tf.add_n(kl_terms) if kl_terms else 0.0
        tf.summary.scalar("infoDrop", kl_sum)
        cost+=kl_sum*0.0001

        tvars = [v for v in tf.trainable_variables()]
        for var in tvars:
            name = var.name.replace("var_lengths", "")
            tf.summary.histogram(name + '/histogram', var)

        regvars = [var for var in tvars if "CvK" in var.name]
        print(regvars)
        reg_costlist = [tf.reduce_sum(tf.square(var)) for var in regvars]
        reg_cost = tf.add_n(reg_costlist)
        tf.summary.scalar("base/regularize_loss", reg_cost)

        # optimizer

        self.local_lr = self.learning_rate

        optimizer = RAdamOptimizer(self.local_lr, epsilon=1e-5, L2_decay=0.01, L1_decay=0.00, decay_vars=regvars,
                                   total_steps=cnf.training_iters,
                                   warmup_proportion=cnf.num_warmup_steps / cnf.training_iters, clip_gradients=True)

        self.optimizer = optimizer.minimize(cost, global_step=self.global_step, colocate_gradients_with_ops=True)

        # some values for printout
        max_vals = []

        for var in tvars:
            var_v = optimizer.get_slot(var, "v")
            max_vals.append(tf.sqrt(var_v))

        self.gnorm = tf.global_norm(max_vals)
        tf.summary.scalar("base/gnorm", self.gnorm)
        self.cost_list = tf.stack(self.cost_list)

    def prepare_dict(self, batch_xs_list, batch_ys_list):
        """Prepares a dictionary of input output values for all bins to do training"""
        feed_dict = {}
        for x_in, data_x in zip(self.x_input, batch_xs_list):
            feed_dict[x_in.name] = data_x
        for y_in, data_y in zip(self.y_input, batch_ys_list):
            feed_dict[y_in.name] = data_y

        return feed_dict

    def prepare_test_dict(self, batch_xs_list, batch_ys_list):
        """Prepares a dictionary of input output values for all bins to do testing"""
        feed_dict = {}
        feed_dict[self.test_x.name] = batch_xs_list[0]
        feed_dict[self.test_y.name] = batch_ys_list[0]
        return feed_dict

    def get_all_mem(self, sess, batch_xs_list, batch_ys_list):
        """Gets an execution trace for the given inputs"""
        feed_dict = self.prepare_test_dict(batch_xs_list, batch_ys_list)
        mem = sess.run(self.allMem, feed_dict=feed_dict)
        return mem

    def get_accuracy(self, sess, batch_xs_list, batch_ys_list):
        """Gets accuracy on the given test examples"""
        feed_dict = self.prepare_test_dict(batch_xs_list, batch_ys_list)
        acc, result, summary = sess.run((self.test_accuracy, self.result, self.test_summary), feed_dict=feed_dict)
        return acc, result, summary

    def get_result(self, sess, batch_xs_list, batch_ys_list):
        """For musicnet. Gets flat labels/predictions on the given test examples"""
        feed_dict = self.prepare_test_dict(batch_xs_list, batch_ys_list)
        result = sess.run(self.result, feed_dict=feed_dict)
        return result

    def get_learning_rate(self, sess):
        rate = sess.run(self.local_lr)
        return rate

    def print_loss(self, sess, batch_xs_list, batch_ys_list):
        """prints training loss on the given inputs"""
        feed_dict = self.prepare_dict(batch_xs_list, batch_ys_list)
        acc, loss, costs, norm11, regul, beta2, summaries = sess.run((self.accuracy, self.base_cost, self.cost_list,
                                                                      self.gnorm, self.sat_loss, self.beta2_rate,
                                                                      self.variable_summaries),
                                                                     feed_dict=feed_dict)
        print("Loss= " + "{:.6f}".format(loss) + \
              ", Accuracy= " + "{:.6f}".format(acc), costs, "gnorm=", norm11, "saturation=", regul)
        return summaries

    def train(self, sess, batch_xs_list, batch_ys_list):
        """do training"""
        feed_dict = self.prepare_dict(batch_xs_list, batch_ys_list)

        res = sess.run([self.base_cost, self.optimizer, self.accuracy, self.cost_list, self.sat_loss] + self.bin_losses,
                       feed_dict=feed_dict)
        loss = res[0]
        acc = res[2]
        costs = res[3]
        regul = res[4]
        loss_per_item = res[5:]
        return loss, acc, loss_per_item, costs, regul

    def set_saturation_weight(self, sess, koef):
        cur_val = sess.run(self.saturation_weight)
        cur_learning_rate = sess.run(self.local_lr)
        koef *= cur_val * cur_learning_rate
        koef = max(min(koef, 1e-3), 1e-20)
        sess.run(self.assign_saturation_weight_op, feed_dict={self.tmpfloat: koef})
