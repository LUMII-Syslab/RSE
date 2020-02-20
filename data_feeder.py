"""For feeding the data to the model"""

import random

import config as cnf
import data_utils as data_gen


class DataSupplier:

    def supply_training_data(self, length, batch_size) -> tuple:
        pass

    def supply_validation_data(self, length, batch_size) -> tuple:
        pass

    def supply_test_data(self, length, batch_size) -> tuple:
        pass


class DefaultSupplier(DataSupplier):
    def supply_training_data(self, length, batch_size) -> tuple:
        return self.__gen_training_data(True)

    def supply_validation_data(self, length, batch_size) -> tuple:
        return self.__gen_training_data(False)

    @staticmethod
    def __gen_training_data(for_training):
        x = []
        y = []

        for index, seq_len in enumerate(cnf.bins):
            data, labels = data_gen.get_batch(seq_len, cnf.batch_size, for_training, cnf.task)
            x += [data]
            y += [labels]

        return x, y

    def supply_test_data(self, length, batch_size):
        data, labels = data_gen.get_batch(length, batch_size, False, cnf.task)
        return [data], [labels]


def create_batch(generator, batch_size, length, for_training=False):
    qna = []
    while len(qna) < batch_size:
        question, answer = next(generator)
        if max(len(question), len(answer)) > length:
            continue

        question_and_answer = data_gen.add_padding(question, answer, length)
        qna.append(question_and_answer)

    random.shuffle(qna)
    questions, answers = tuple(zip(*qna))
    return [questions], [answers]


def create_data_supplier() -> DataSupplier:
    return DefaultSupplier()
