import os
import pickle
import tarfile
from typing import List

import tensorflow as tf

import config as cnf
import data_utils as data_utils
import language.utils as utils
from language.utils import LanguageTask

lambada_data_set = "http://clic.cimec.unitn.it/lambada/lambada-dataset.tar.gz"
lambada_train_set = "http://ttic.uchicago.edu/~kgimpel/data/lambada-train-valid.tar.gz"

PADDING = "C_PAD"
UNKNOWN = "C_UNK"
UNKNOWN_ID = 1
SELECTED = "C_SEL"
SELECTED_ID = 2
MASK_TOKEN = "C_MASK"
MASK_TOKEN_ID = 3


def lambada_file(file: str) -> str:
    return os.path.join(cnf.lambada_data_dir, file)


train_file = lambada_file("train.txt")
test_file = lambada_file("lambada_test_plain_text.txt")
vocab_file = lambada_file("vocab_lambada_{size}.txt".format(size=cnf.lambada_vocab_size))
test_token_file = lambada_file("test_tokens_{size}.txt".format(size=cnf.lambada_vocab_size))
train_token_file = lambada_file("train_tokens_{size}.txt".format(size=cnf.lambada_vocab_size))


def download_lambada():
    l_data = utils.download(cnf.lambada_data_dir, "lambada_data_set.tar.gz", lambada_data_set)
    utils.extract_tar(l_data, cnf.lambada_data_dir)

    l_train = utils.download(cnf.lambada_data_dir, "lambada_train_set.tar.gz", lambada_train_set)
    with tarfile.open(l_train, "r") as tar:
        print("Opening Lambada archive")
        train_set = tar.getmember("lambada-train-valid/" + "train.txt")
        train_set.name = "train.txt"
        tar.extract(train_set, cnf.lambada_data_dir)


def read_file(file_name: str) -> list:
    with open(file_name, "r", encoding="utf-8") as file:
        return file.readlines()


def load_embedding_vocabulary() -> dict:
    if not tf.gfile.Exists(cnf.emb_word_dictionary):
        utils.prepare_embeddings()

    with open(cnf.emb_word_dictionary, "rb") as dict_file:
        return pickle.load(dict_file)


def tokenize_files():
    if not tf.gfile.Exists(train_file):
        download_lambada()

    lines = read_file(train_file)  # Tokenize train file

    print("Tokenizing Lambada data")

    if cnf.use_pre_trained_embedding:
        vocab = load_embedding_vocabulary()
    else:
        vocab = prepare_custom_vocabulary(lines)

    with open(train_token_file, "w", encoding="utf-8") as file:
        for line in lines:
            line_tokens = str(line).split()
            tokens = []
            for word in line_tokens:
                tokens.append(str(vocab.get(word, 1)))
            file.write(" ".join(tokens) + "\n")

    lines = read_file(test_file)  # Tokenize test file

    with open(test_token_file, "w", encoding="utf-8") as file:
        for line in lines:
            line_tokens = str(line).split()
            tokens = []
            for word in line_tokens:
                tokens.append(str(vocab.get(word, 1)))  # Unknown word otherwise
            file.write(" ".join(tokens) + "\n")


def prepare_custom_vocabulary(lines):
    vocab = {}
    for line in lines:
        line_tokens = str(line).split()
        for word in line_tokens:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    sort = sorted(vocab, key=vocab.get, reverse=True)
    sort = [PADDING, UNKNOWN, SELECTED, MASK_TOKEN] + sort
    sort = sort[:cnf.lambada_vocab_size]
    with open(vocab_file, "w", encoding="utf-8") as file:
        for line in sort:
            file.write(line + "\n")
    vocab = {value: index for index, value in enumerate(sort)}
    return vocab


class LambadaTask(LanguageTask):

    def prepare_train_data(self):
        if not tf.gfile.Exists(train_token_file):
            tokenize_files()

        lines = read_file(train_token_file)
        self._create_cases_for_lines(lines, data_utils.train_set)

    @staticmethod
    def _create_cases_for_lines(lines: list, case_set: dict, is_test=False) -> None:
        for line in lines:
            in_data = [int(token) for token in line.split()]
            answer = in_data[-1]  # Last word is answer on Lambada data set
            in_data[-1] = MASK_TOKEN_ID  # Mask last word

            if not is_test and (answer not in in_data):
                continue  # Ignore cases where answer not in text

            out_data = [SELECTED_ID if word == answer else 1 for word in in_data]
            length = len(in_data)

            case_set["lambada"][length].append([in_data, out_data])

    def prepare_test_data(self):

        if not tf.gfile.Exists(test_token_file):
            tokenize_files()

        lines = read_file(test_token_file)
        self._create_cases_for_lines(lines, data_utils.test_set, is_test=True)

    def prepare_data(self):
        print("Preparing LAMBADA training data")
        self.prepare_train_data()
        print("Prepering LAMBADA test data")
        self.prepare_test_data()

    def detokenizer(self):
        with open(vocab_file, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()

        vocab = {index: value for index, value in enumerate(lines)}
        return Detokenizer(vocab)


class Detokenizer:
    def __init__(self, vocab: dict) -> None:
        self._vocab = vocab

    def detokenize_batch(self, batch: list) -> list:
        return [self.detokenize_sentence(sen) for sen in batch]

    def detokenize_sentence(self, tokens) -> List[str]:
        sentence = [self.detokenize_token(token) for token in tokens]
        return sentence

    def _padding_position(self, sentence):
        try:
            return sentence.index(PADDING)
        except ValueError:
            return len(sentence)

    def detokenize_token(self, token) -> str:
        return self._vocab.get(token) if token in self._vocab else self._vocab.get(UNKNOWN)


class LambadaTaskWord(LambadaTask):

    @staticmethod
    def _create_cases_for_lines(lines: list, case_set: dict, is_test=False) -> None:
        for line in lines:
            in_data = [int(token) for token in line.split()]
            answer = in_data[-1]  # Last word is answer on Lambada data set
            in_data[-1] = MASK_TOKEN_ID  # Mask last word

            if not is_test and (answer not in in_data):
                continue  # Ignore cases where answer not in text

            out_data = [answer]
            length = len(in_data)

            case_set["lambada_w"][length].append([in_data, out_data])
