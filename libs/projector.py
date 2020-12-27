#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Tian Tian
@File    : projector.py   
@Time    : 2020/11/4 2:10 PM
@Desc    : 用于启动 tensorboard embedding projector，展示字符的 embedding
@Version : 1.0

tensorboard embbeding projector 的生成，可参考 https://www.cnblogs.com/cloud-ken/p/9329703.html
"""

import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
from tensorboard.plugins import projector

import libs.utils as utils
from libs.predictor import CrnnEmbeddingPredictor

SPRITE_FILE = 'crnn_sprite.jpg'
META_FILE = "crnn_meta.tsv"
TENSOR_NAME = "LSTM_out"


class Projector:
    def __init__(self, predictor, model, charset, file):
        self.predictor = predictor
        self.predictor.tensor_define(model_path=model, charset_path=charset, label_file=file)
        # 只想展示部分字符的 embedding 时使用，例如 self.pick=['自','白']
        self.pick = []

    def one_batch_pocess(self, img_b, label_b):
        """
        Process one batch embedding
        Args:
            img_b: list of imgs
            label_b: list of label strings

        Returns:
            lstm_out: embedding vectors to show
            imgs: small imgs of chars to show
            char_single: labels of small imgs

        """
        inputs = self.predictor.make_inputs(img_b, label_b)
        chars, char_pos, char_segs, lstm_out = self.predictor.pred(inputs)

        img_origin = img_b * 128 + 128
        imgs = self.predictor.cut_single_all_char(img_origin, char_segs)
        imgs = utils.resize_batch_image_single(imgs, 32, 32)

        char_single = []
        for char in chars:
            for s in char:
                char_single.append(s)

        return lstm_out, imgs, char_single

    def pick_specific_char(self, embeddings, imgs, labels):
        """
        filter to show only interested chars
        Args:
            embeddings: embedding vectors
            imgs: imgs of chars
            labels: labels of small imgs

        Returns:
            after filtering

        """
        if not self.pick:
            return embeddings, imgs, labels

        emb_picked, label_picked, img_picked = [], [], []
        for e, l, i in zip(embeddings, labels, imgs):
            if l in self.pick:
                emb_picked.append(e)
                label_picked.append(l)
                img_picked.append(i)

        return emb_picked, img_picked, label_picked

    def process_embedding(self):
        """
        Process all batches embedding

        Returns:
            lstm_out: embedding vectors to show
            imgs: small imgs of chars to show
            char_single: labels of small imgs

        """

        lstm_out, imgs, labels = [], [], []
        dataset = self.predictor.dataset
        for batch in range(dataset.num_batches):
            try:
                img_batch, label_batch, batch_labels, batch_img_paths = dataset.get_next_batch(self.predictor.sess)

                _lstm_out, _imgs, _preds = self.one_batch_pocess(img_batch, label_batch)
                char_num = _lstm_out.shape[0]
                _lstm_out = np.split(_lstm_out, char_num)

                _e, _i, _l = self.pick_specific_char(_lstm_out, _imgs, _preds)

                lstm_out += _e
                imgs += _i
                labels += _l
            except Exception:
                import traceback
                traceback.print_exc()
                print('处理bacth发生错误，跳过本 batch ')

        lstm_out = np.concatenate(lstm_out)

        return lstm_out, imgs, labels

    @staticmethod
    def create_sprite_image(imgs):
        """
        Return a so-called sprite img
        Further information: https://www.cnblogs.com/cloud-ken/p/9329703.html
        Args:
            imgs: small imgs of single chars

        Returns: sprite img used by tensorboard

        """
        if isinstance(imgs, list):
            imgs = np.array(imgs)
        img_h = imgs.shape[1]
        img_w = imgs.shape[2]
        img_c = imgs.shape[3]

        # sprite 图像可以理解成是小图片拼成的大正方形矩阵，大正方形矩阵中的每一个元素就是原来的小图片
        # 于是这个正方形的边长就是 sqrt(n),其中n为小图片的数量
        n_plots = int(np.ceil(np.sqrt(imgs.shape[0])))

        # 初始化大图片
        sprite_image = np.ones((img_h * n_plots, img_w * n_plots, img_c))

        for i in range(n_plots):
            for j in range(n_plots):
                # 计算当前图片的编号
                this_filter = i * n_plots + j
                if this_filter < imgs.shape[0]:
                    # 将当前小图片的内容复制到大图中
                    this_img = imgs[this_filter]
                    sprite_image[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w, :] = this_img

        save_path = os.path.join(LOG_DIR, SPRITE_FILE)
        cv2.imwrite(save_path, sprite_image)

        return sprite_image

    @staticmethod
    def create_meta_data(labels):
        """
        Create meta data as tsv file
        Looks like:
        Index   Label
        0   我
        1   爱
        2   北

        Args:
            labels:

        Returns:

        """
        save_path = os.path.join(LOG_DIR, META_FILE)
        with open(save_path, 'w') as f:
            f.write("Index\tLabel\n")
            for i, label in enumerate(labels):
                f.write("%d\t%s\n" % (i, label))

    def visualisation(self):
        """
        Main process of visualisation
        Returns:

        """
        # get embedding_arrays, sprite_images, meta_data
        embedding, imgs, labels = self.process_embedding()
        self.create_sprite_image(imgs)
        self.create_meta_data(labels)

        emb = tf.Variable(embedding, name=TENSOR_NAME)

        # embedding configs
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = emb.name

        embedding.metadata_path = META_FILE
        embedding.sprite.image_path = SPRITE_FILE
        embedding.sprite.single_image_dim.extend([32, 32])

        # save data
        summary_writer = tf.summary.FileWriter(LOG_DIR)
        projector.visualize_embeddings(summary_writer, config)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(LOG_DIR, "emb"), self.predictor.global_step)

        summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='./output_20200918/checkpoint/default/ctc_center',
                        help="model used to infer embeddings")
    parser.add_argument("--file", type=str, default='./data_example/train_temp.txt', help="label file")
    parser.add_argument("--dir", type=str, default='output_20200918/emb_log', help="directory of tensorboard")

    a = parser.parse_args()

    LOG_DIR = a.dir

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    embedding_maker = CrnnEmbeddingPredictor(cfg_name='resnet')
    proj = Projector(embedding_maker, model=a.model, charset='./data/chars/lexicon.txt', file=a.file)
    proj.visualisation()