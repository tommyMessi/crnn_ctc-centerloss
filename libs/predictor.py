#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Tian Tian
@File    : predictor.py   
@Time    : 2020/10/9 11:40 AM
@Desc    : 加载模型，预测给定的 tensor 的数值
@Version : 1.0 
"""
import os
import tensorflow as tf
import logging

from nets.crnn import CRNN
from libs.label_converter import LabelConverter
from libs.config import load_config
from libs import utils
from libs.img_dataset import ImgDataset

logger = logging.getLogger(__name__)


class BasePredictor:
    def __init__(self, cfg_name):
        self.cfg = load_config(cfg_name)
        self.cfg.lr_boundaries = [10000]
        self.cfg.lr_values = [self.cfg.lr * (self.cfg.lr_decay_rate ** i) for i in
                              range(len(self.cfg.lr_boundaries) + 1)]
        self.sess = None
        self.graph = None
        self.converter = None
        self.dataset = None
        self.input = None
        self.output = None
        self.global_step = None

    @staticmethod
    def load_model(model_path, sess):
        res_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        res_vars = [v for v in res_vars if v.name.find('centers') == -1]
        saver = tf.train.Saver(res_vars)
        if os.path.exists(model_path + ".index"):
            logger.debug("恢复给定名字的模型：%s", model_path)
            saver.restore(sess, model_path)
            return sess

        else:
            logger.error("模型不存在！：[%s]", model_path)
            return None

    def make_inputs(self, images, labels=None):
        """
        从原始数据得到模型的输入数据
        Args:
            images:
            labels:

        Returns:

        """
        pass

    def tensor_define(self, model_path, charset_path, label_file):
        """
        定义模型、输入、输出
        Args:
            model_path:
            charset_path:
            label_file:

        Returns:

        """
        pass

    def tensor_collect(self, inputs):
        """
        得到模型计算结果
        Args:
            inputs:

        Returns:

        """
        feed_dict = {tensor: data_in for tensor, data_in in zip(self.input, inputs)}
        tensor_list = self.output

        result = self.sess.run(tensor_list, feed_dict=feed_dict)

        return result

    def pred(self, inputs):
        """
        模型预测，包括计算结果后处理
        Args:
            inputs:

        Returns:

        """
        pass


class CrnnEmbeddingPredictor(BasePredictor):
    def make_inputs(self, images, labels=None):
        single_label = labels[1]
        pos_init = [[-1, -1]]
        w = utils.round_up(images.shape[2] / 4)
        charnum_pseudo = [1]

        return images, labels, single_label, w, charnum_pseudo, pos_init, False

    def tensor_define(self, model_path, charset_path, label_file):

        converter = LabelConverter(chars_file=charset_path)
        dataset = ImgDataset(label_file, converter, batch_size=1, shuffle=False)
        model = CRNN(self.cfg, num_classes=converter.num_classes)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(dataset.init_op)

        sess = self.load_model(model_path=model_path, sess=sess)
        global_step = sess.run(model.global_step)

        self.sess = sess
        self.converter = converter
        self.dataset = dataset

        self.input = [model.inputs, model.labels, model.bat_labels, model.len_labels, model.char_num,
                      model.char_pos_init, model.is_training]
        self.output = [model.dense_decoded, model.char_pos, model.embedding]
        self.global_step = global_step

    def pred(self, inputs):
        decodes, char_pos, lstm_out = self.tensor_collect(inputs)
        char_segs = utils.get_char_segment(char_pos)

        predicts = [self.converter.decode(p, CRNN.CTC_INVALID_INDEX) for p in decodes]

        return predicts, char_pos, char_segs, lstm_out

    @staticmethod
    def cut_single_one_img(img, char_segs):
        """
        单张图片切字
        Args:
            img:
            char_segs:

        Returns:

        """
        char_imgs = []

        for i, seg in enumerate(char_segs):
            img_one_char = img[:, seg, :]
            char_imgs.append(img_one_char)

        return char_imgs

    def cut_single_all_char(self, imgs, all_segs):
        """
        从图片中按照计算好的范围切字
        Args:
            imgs: ndarray of shape (n, h, w, c)
            all_segs: list of lists of slice objects

        Returns:

        """
        single_imgs = []
        # imgs = imgs.split(imgs, imgs.shape[0])

        for i, img_segs in enumerate(all_segs):
            img = imgs[i, :, :, :]

            char_imgs = self.cut_single_one_img(img=img, char_segs=img_segs)
            single_imgs += char_imgs

        return single_imgs