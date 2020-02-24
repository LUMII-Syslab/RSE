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

"""Algorithmic tasks"""
import random
import sys

import numpy as np

import config as cnf
import data_utils
from RSE_network import rol


class Task:

    def input_output_pair(self, length) -> tuple:
        """
        :param length: Max length of input
        :return: returns tuple of (input, output)
        """
        pass


def flatten(list_of_list, delimiter):
    res = []

    for x in list_of_list:
        for y in x:
            res.append(y)
        if delimiter:
            res.append(delimiter)

    return res[0:-1]


def generate_probability(length, last_probability):
    t_len = length - 1
    res = np.full(t_len, (1 - last_probability) / t_len)
    return np.append(res, last_probability)


class WordSorting(Task):
    """
    Task for word sorting (w_sort). Outputs words in one list separated by delimiter
    """

    def __init__(self, alphabet: list):
        """
        :param alphabet: Alphabet used. Last element is used as delimiter.
        """
        a_len = len(alphabet)

        self.alphabet = alphabet
        self.delimiter = alphabet[a_len - 1]
        self.probability = generate_probability(a_len, 1 / a_len)

    def _gen_from_alphabet(self, length):
        return [random.choice(self.alphabet) for _ in range(length)]

    def input_output_pair(self, length) -> tuple:
        """
          :param length: length of input/output pair.
          :return: input output pair. Output is sorted.
          """

        def is_even(x_list: list) -> list:
            return [x & 1 for x in x_list]

        word_list = self.gen_word_list(length)
        sorted_word_list = sorted(word_list, key=is_even)
        return flatten(word_list, self.delimiter), flatten(sorted_word_list, self.delimiter)

    def gen_word_list(self, length) -> list:
        """
          :param length: length of input/output pair.
          :return: generated word list
          """
        tmp_rand = np.random.choice(self.alphabet[:-1], length)
        probability = generate_probability(len(self.alphabet), min(0.5, 1.0 / np.sqrt(length)))
        word_dividers = np.random.choice(self.alphabet, length, p=probability)

        i = 1
        while i < length - 1:
            if word_dividers[i] == self.delimiter:
                tmp_rand[i] = self.delimiter
                i += 2
            else:
                i += 1

        word_list = []
        word = []
        for tmp in tmp_rand:
            if tmp == self.delimiter:
                word_list.append(word)
                word = []
            else:
                word.append(tmp)
        if word or (tmp_rand and tmp_rand[-1] == self.delimiter and len(tmp_rand) > 1):  # Add if last word is empty
            word_list.append(word)

        return word_list


class SortedMerge(Task):
    """
    Merging two sorted list of words.
    """

    def __init__(self, alphabet: list):
        """
        :param alphabet: Alphabet used. Last element is used as delimiter. Should be at least 5
        """
        a_len = len(alphabet)
        self.delimiter = alphabet[a_len - 1]
        self.delimiter1 = alphabet[a_len - 2]
        self.word_gen = WordSorting(alphabet[:-1])

    def input_output_pair(self, length) -> tuple:
        """
          :param length: length of input/output pair.
          :return: input output pair. Output is sorted.
          """
        word_list = self.word_gen.gen_word_list(max(1, length - 2))
        n = len(word_list)
        list1 = sorted(word_list[:n // 2])
        list2 = sorted(word_list[n // 2:])
        word_list = list1 + [[self.delimiter1]] + list2
        sorted_word_list = sorted(word_list)
        return flatten(word_list, self.delimiter), flatten(sorted_word_list, self.delimiter)


def test_word_sorting():
    task = WordSorting([x for x in range(1, 100)])
    print(task.input_output_pair(8))


def join_lists_alternately(list1, list2):
    res_list = []
    for i in range(len(list2)):
        res_list.append(list1[i])
        res_list.append(list2[i])
    return res_list


def index_word_list(word_list: list, id_range: int) -> tuple:
    """
    Indexes word characters.
    :return: cur_ind - one index after last character in last word
             enum-list - pair of list of lists tuples, where first element list of index and second element - word
    """

    ids = [i for i in range(1, id_range + 1)]
    indexed_list = [(np.random.choice(ids, len(word)).tolist(), word) for word in word_list]
    return np.random.choice(ids), indexed_list


def reverse_bit(x, n):
    result = 0
    for i in range(n):
        if (x >> i) & 1: result |= 1 << (n - 1 - i)
    return result


def to_base(num, b, l=1):
    assert num >= 0
    ans = []
    while num:
        ans.append(num % b)
        num //= b
    while len(ans) < l:
        ans.append(0)
    return ans


def tobcd(num):
    res = []
    for digit in num:
        bin_digit = to_base(digit, 2, 4)
        bin_digit[3] += 2  # digit end marker
        res += bin_digit
    return res


def add(n1, n2, base=10):
    """Add two numbers represented as lower-endian digit lists."""
    k = max(len(n1), len(n2)) + 1
    d1 = n1 + [0 for _ in range(k - len(n1))]
    d2 = n2 + [0 for _ in range(k - len(n2))]
    res = []
    carry = 0
    for i in range(k):
        if d1[i] + d2[i] + carry < base:
            res.append(d1[i] + d2[i] + carry)
            carry = 0
        else:
            res.append(d1[i] + d2[i] + carry - base)
            carry = 1
    while res and res[-1] == 0:
        res = res[:-1]
    if res: return res
    return [0]


class Default(Task):

    def __init__(self, task: str, max_input_digit: int):
        self.task = task
        self.max_input_digit = max_input_digit

    def input_output_pair(self, length) -> tuple:
        if self.task in ["add", "badd", "qadd", "bmul", "mul", "qmul", "mulbcd"]:
            return self.rand_pair(length)
        elif self.task == "dup":
            return self.rand_dup_pair(length, self.max_input_digit)
        elif self.task == "rev2":
            return self.rand_rev2_pair(length, self.max_input_digit)
        elif self.task == "search":
            return self.rand_search_pair(length, self.max_input_digit)
        elif self.task == "kvsort":
            return self.rand_kvsort_pair(length, self.max_input_digit)
        elif self.task == "div":
            return self.rand_div_pair(length)
        else:
            i = [np.random.randint(self.max_input_digit - 1) + 1 for _ in range(length)]
            t = self.spec(i, self.task, self.max_input_digit)
            return i, t

    def rand_pair(self, length):
        task = self.task
        """Random data pair for a task. Total length should be <= l."""
        k = (length - 1) // 2
        if task == "mulbcd": k = (length - 1) // 8
        base = 10
        if task[0] == "b": base = 2
        if task[0] == "q": base = 4
        d1 = [random.randrange(base) for _ in range(k)]
        d2 = [random.randrange(base) for _ in range(k)]
        #d2 = [np.asscalar(num) for num in np.random.randint(base, size=k)]
        if task in ["add", "badd", "qadd"]:
            res = add(d1, d2, base)
        elif task in ["mul", "bmul", "qmul", "mulbcd"]:
            d1n = sum([d * (base ** i) for i, d in enumerate(d1)])
            d2n = sum([d * (base ** i) for i, d in enumerate(d2)])
            if task == "bmul":
                res = to_base(d1n * d2n, base, k * 2 + 1)
            elif task == "mul":
                res = [int(x) for x in list(reversed(str(d1n * d2n)))]
            elif task == "qmul":
                res = to_base(d1n * d2n, base, k * 2 + 1)
            elif task == "mulbcd":
                res = to_base(d1n * d2n, base, k * 2)
                res = tobcd(res)
                d1 = tobcd(d1)
                d2 = tobcd(d2)
        else:
            sys.exit()
        sep = [12]
        if task in ["add", "badd", "qadd"]: sep = [11]
        inp = [d + 1 for d in d1] + sep + [d + 1 for d in d2]
        return inp, [r + 1 for r in res]

    def rand_div_pair(self, length):
        if length < 3: return [], []
        base = 2
        k = (length - 1) // 2
        k1 = random.randrange(1, k + 1)
        assert k1 > 0
        assert k > 0
        assert k >= k1
        while True:
            d1 = [np.asscalar(num) for num in np.random.randint(base, size=k)]
            d2 = [np.asscalar(num) for num in np.random.randint(base, size=k1)]
            d1n = sum([d * (base ** i) for i, d in enumerate(d1)])
            d2n = sum([d * (base ** i) for i, d in enumerate(d2)])
            if d2n > 0: break
        res0 = to_base(d1n // d2n, base, k)
        res1 = to_base(d1n % d2n, base, k)
        d2 = to_base(d2n, base, k)
        sep = [12]
        sep_res = [5]
        inp = [d + 1 for d in d1] + sep + [d + 3 for d in d2]
        res = [d + 1 for d in res0] + sep_res + [d + 3 for d in res1]
        assert len(res) <= length
        assert len(inp) == len(res)
        return inp, res

    def rand_dup_pair(self, l, nclass):
        """Random data pair for duplication task. Total length should be <= l."""
        k = l // 2
        x = [np.random.randint(nclass - 1) + 1 for _ in range(k)]
        inp = x + [0 for _ in range(l - k)]
        res = x + x + [0 for _ in range(l - 2 * k)]
        return inp, res

    def rand_rev2_pair(self, l, nclass):
        """Random data pair for reverse2 task. Total length should be <= l."""
        inp = [(np.random.randint(nclass - 1) + 1,
                np.random.randint(nclass - 1) + 1) for _ in range(l // 2)]
        res = [i for i in reversed(inp)]
        return [x for p in inp for x in p], [x for p in res for x in p]

    def rand_search_pair(self, l, nclass):
        """Random data pair for search task. Total length should be <= l."""
        inp = [(np.random.randint(nclass - 1) + 1,
                np.random.randint(nclass - 1) + 1) for _ in range((l - 1) // 2)]
        q = np.random.randint(nclass - 1) + 1
        res = 0
        for (k, v) in reversed(inp):
            if k == q:
                res = v
        return [x for p in inp for x in p] + [q], [res]

    def rand_kvsort_pair(self, l, nclass):
        """Random data pair for key-value sort. Total length should be <= l."""
        keys = [(np.random.randint(nclass - 1) + 1, i) for i in range(l // 2)]
        vals = [np.random.randint(nclass - 1) + 1 for _ in range(l // 2)]
        kv = [(k, vals[i]) for (k, i) in keys]
        sorted_kv = [(k, vals[i]) for (k, i) in sorted(keys)]
        return [x for p in kv for x in p], [x for p in sorted_kv for x in p]

    def spec(self, inp, task, nclass):
        """Return the target given the input for some tasks."""
        if task == "sort":
            return sorted(inp)
        elif task == "id":
            return inp
        elif task == "rev":
            return [i for i in reversed(inp)]
        elif task == "shuffle":  # bit reverse permutation
            n_bits = (len(inp) - 1).bit_length()
            res = []
            for i in range(len(inp)):
                i1 = reverse_bit(i, n_bits) % len(inp)
                res.append(inp[i1])
            return res
        elif task == "incr":
            carry = 1
            res = []
            for i in range(len(inp)):
                if inp[i] + carry < nclass:
                    res.append(inp[i] + carry)
                    carry = 0
                else:
                    res.append(1)
                    carry = 1
            return res
        elif task == "left":
            return [inp[0]]
        elif task == "right":
            return [inp[-1]]
        elif task == "left-shift":
            return [inp[l - 1] for l in range(len(inp))]
        elif task == "right-shift":
            return [inp[l + 1] for l in range(len(inp))]
        else:
            data_utils.print_out("Unknown spec for task " + str(task))
            sys.exit()


def input_alphabet():
    return [x for x in range(1, cnf.n_input)]


def select_task(task_name: str, max_input_digit: int) -> Task:
    if task_name == "memory_indexing":
        return MemoryIndexing(max_input_digit)
    elif task_name == "rol":
        return ShiftTask(max_input_digit)
    elif task_name == "merge":
        return SortedMerge(input_alphabet())
    elif task_name == "w_sort":
        return WordSorting(input_alphabet())
    elif task_name == "dyck":
        return Dyck(max_input_digit)
    elif task_name == "dyck_continue":
        return DyckLastBracket(max_input_digit)
    else:
        return Default(task_name, max_input_digit)


class Dyck(Task):

    def __init__(self, max_val) -> None:
        self.bracket_pairs = {i: i + 1 for i in range(1, max_val, 2)}
        self.inverse_bracket_pairs = {v: k for k, v in self.bracket_pairs.items()}  # inverse
        self.open_brackets = [k for k, _ in self.bracket_pairs.items()]
        self.brackets = [k for k, _ in self.bracket_pairs.items()] + [v for _, v in self.bracket_pairs.items()]

    def generate_dyck_word(self, length):
        word = []
        stack = []

        fix = length % 2
        while len(word) + len(stack) + fix < length:
            possible_brackets = [] + self.open_brackets
            if stack:
                possible_brackets.append(stack[-1])

            br = random.choice(possible_brackets)
            word.append(br)

            if stack and br == stack[-1]:
                stack.pop()
            else:
                stack.append(self.bracket_pairs.get(br))

        while stack:
            word.append(stack.pop())

        return word

    def _is_dyck(self, word):
        stack = []
        pairs = self.inverse_bracket_pairs

        for br in word:
            if stack and br in pairs and stack[-1] == pairs.get(br):
                stack.pop()
            else:
                stack.append(br)

        return not stack

    def _generate_not_dyck_word(self, length):
        word = self.generate_dyck_word(length)
        while True:
            count = random.randint(1, 4)
            for _ in range(count):
                pos = random.randint(0, length - (length % 2) - 1)
                word[pos] = random.choice(self.brackets)
            if not self._is_dyck(word):
                return word

    def _generate_random_word(self, length):
        while True:
            word = np.random.choice(self.brackets, length)
            if not self._is_dyck(word):
                return word

    def input_output_pair(self, length) -> tuple:

        word, is_dyck = self._generate_word(length)
        return [int(ch) for ch in word], is_dyck

    def _generate_word(self, length):
        choice = np.random.choice([0, 1])

        if length == 1:
            return self._generate_random_word(length), [1]
        elif choice == 1:
            return self._generate_not_dyck_word(length), [1]
        else:
            return self.generate_dyck_word(length), [2]


class DyckLastBracket(Task):

    def __init__(self, max_length) -> None:
        self.dyck = Dyck(max_length)

    def generate_case(self, length) -> tuple:
        word = []
        stack = []

        while len(word) < length:
            possible_brackets = [] + self.dyck.open_brackets
            if stack:
                possible_brackets.append(stack[-1])

            br = random.choice(possible_brackets)

            if stack and br == stack[-1]:
                if len(word) + 1 < length:
                    word.append(br)
                    stack.pop()
            else:
                word.append(br)
                stack.append(self.dyck.bracket_pairs.get(br))

        return word, list(reversed(stack))

    def input_output_pair(self, length) -> tuple:
        return self.generate_case(length)


def test_dyck_last_bracket():
    br = DyckLastBracket(5)
    for _ in range(100):
        print(br.input_output_pair(5))


class ShiftTask(Task):
    """
    Task for bitwise cyclic rotation of binary numbers.
    shift amount is generated in unary
    """

    def __init__(self, n_class) -> None:
        self.n_class = n_class
        self.base = n_class - 3

    def generate_case(self, length) -> tuple:
        if length < 2: return [], []
        number_length = length // 2
        number = random.getrandbits(number_length)
        shift_amount = random.randrange(number_length)
        shifted = rol(number, number_length, shift_amount)

        number_str = to_base(number, self.base, number_length)
        out_str = to_base(shifted, self.base, number_length)
        shift_unary = [self.n_class - 1] * shift_amount + [self.n_class - 2] * (number_length - 1 - shift_amount)
        x = [a + 1 for a in number_str] + shift_unary
        return x, [a + 1 for a in out_str]

    def input_output_pair(self, length) -> tuple:
        x, y = self.generate_case(length)
        return x, y


class MemoryIndexing(Task):
    """
    Retrieves values from from memory by their keys.
    Input consists of key-value pairs and a list of keys to be retrieved
    output have the list of values corresponding to the keys
    """

    def __init__(self, n_class) -> None:
        self.n_class = n_class
        self.key_length = (n_class - 1) // 3
        self.key_list = list(range(self.key_length))

    def generate_case(self, length) -> tuple:
        if length < 2: return [], []
        number_length = length // 2
        random.shuffle(self.key_list)
        assert self.key_length >= number_length  # there should be a lot of keys
        keys = self.key_list[:number_length]
        data = np.random.randint(2, size=[number_length])
        key_vals = [k * 3 + d for k, d in zip(keys, data)]
        indices = [k * 3 + 2 for k in keys]
        random.shuffle(key_vals)

        x = key_vals + indices
        return [a + 1 for a in x], [a + 1 for a in data]

    def input_output_pair(self, length) -> tuple:
        x, y = self.generate_case(length)
        return x, y
