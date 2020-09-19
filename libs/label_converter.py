import sys, os
from . import utils

from libs.utils import load_chars


class LabelConverter:
    def __init__(self, chars_file):
        self.chars = ''.join(load_chars(chars_file))
        # char_set_length + ctc_blank
        self.num_classes = len(self.chars) + 1

        self.encode_maps = {}
        self.decode_maps = {}

        self.create_encode_decode_maps(self.chars)

        print('Load chars file: %s num_classes: %d + 1(CTC Black)' % (chars_file, self.num_classes - 1))

    def create_encode_decode_maps(self, chars):
        for i, char in enumerate(chars):
            self.encode_maps[char] = i
            self.decode_maps[i] = char

    def encode(self, label):
        """如果 label 中有字符集中不存在的字符，则忽略"""
        encoded_label = []
        for c in label:
            if c in self.chars:
                encoded_label.append(self.encode_maps[c])
                # else:
                #     encoded_label.append(-1)

        return encoded_label

    def encode_list(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(self.encode(label))
        return encoded_labels

    def decode(self, encoded_label, invalid_index):
        """
        :param encoded_label result of ctc_greedy_decoder
        :param invalid_index ctc空白符的索引
        :return decode label string
        """
        label = []
        for index, char_index in enumerate(encoded_label):
            if char_index != invalid_index:
                label.append(char_index)

        label = [self.decode_maps[c] for c in label]
        return ''.join(label).strip()

    def decode_list(self, encoded_labels, invalid_index):
        decoded_labels = []
        for encoded_label in encoded_labels:
            decoded_labels.append(self.decode(encoded_label, invalid_index))
        return decoded_labels
