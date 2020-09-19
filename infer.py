# coding:utf-8
from __future__ import unicode_literals

import copy
import time
import os
import json
import logging

import numpy as np
import tensorflow as tf

from PIL import Image
from libs.label_converter import LabelConverter
from nets.crnn import CRNN
from libs.config import load_config
from libs.utils import ctc_label
from utils import image_to_pil

lexicon = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/chars/lexicon.txt')

BATCH_SIZE = 32

PAD_IMAGE = Image.new('L', (16, 32), color=255)


class Infer(object):
    def __init__(self, model_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        self.cfg = load_config('resnet')
        self.label_converter = LabelConverter(lexicon)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=config)

        with self.session.as_default():
            with self.graph.as_default():
                self.net = CRNN(self.cfg, num_classes=self.label_converter.num_classes)
                saver = tf.train.Saver()
                saver.restore(self.session, model_path)

        logging.info('CRNN model initialized.')

    def normalize_image(self, img):
        """
        将图像归一化到高为32
        :param img:
        :return:
        """
        img = img.convert('L')
        w, h = img.size
        rio = h / 32.0
        w0 = int(round(w / rio))
        img = img.resize((max(w0, 1), 32), Image.BICUBIC)
        return img

    def predict(self, image, long_info=True):
        """
        单张预测
        :param image:
        :return:
        """
        start_time = time.time()
        image = image_to_pil(image)
        image_width = image.width
        image = self.normalize_image(image)
        if image.width <= 4:
            text = ''
            return text

        image = np.reshape(image, (1, 32, image.width, 1))
        image = (image.astype(np.float32) - 128.0) / 128.0

        feed = {self.net.feeds()['inputs']: image,
                self.net.feeds()['is_training']: False
                }
        predict_label, predict_prob, logits = self.session.run(self.net.fetches(), feed_dict=feed)
        p, weights, positions = ctc_label(predict_label[0], predict_prob[0], image_width,
                                          blank_index=self.label_converter.num_classes - 1)
        txt = self.label_converter.decode(p, invalid_index=-1)
        ret = dict()
        ret['text'] = txt
        ret['weights'] = [float(weight) for weight in weights[:len(txt)]]
        ret['positions'] = [float(position) for position in positions[:len(txt)]]
        ret['direction'] = 0
        print('predict time is %.4f ms' % ((time.time() - start_time) * 1000))
        if long_info:
            return json.dumps(ret, ensure_ascii=False)
        else:
            return ret['text']

    def normalize_batch(self, image_batch):
        """
        将一个batch内的图像归一化到相同的尺寸
        :param image_batch:
        :return:
        """
        input_batch_size = len(image_batch)
        normalized_batch = []
        chars_count = []
        image_width_list = [int(img.width) for img in image_batch]
        batch_image_width = max(image_width_list)
        max_width_image_idx = np.argmax(image_width_list)
        if input_batch_size == BATCH_SIZE:
            for i in range(BATCH_SIZE):
                base_image = copy.deepcopy(image_batch[max_width_image_idx])
                base_image.paste(image_batch[i], (0, 0))
                base_image.paste(PAD_IMAGE, (image_batch[i].width, 0))
                normalized_image = np.reshape(base_image, (32, batch_image_width, 1))
                normalized_image = normalized_image.astype(np.float32) / 128.0 - 1.0
                normalized_batch.append(normalized_image)
                chars_count.append(image_batch[i].width / 4)
        else:
            for i in range(input_batch_size):
                base_image = copy.deepcopy(image_batch[max_width_image_idx])
                base_image.paste(image_batch[i], (0, 0))
                base_image.paste(PAD_IMAGE, (image_batch[i].width, 0))
                normalized_image = np.reshape(base_image, (32, batch_image_width, 1))
                normalized_image = normalized_image.astype(np.float32) / 128.0 - 1.0
                normalized_batch.append(normalized_image)
                chars_count.append(image_batch[i].width / 4)

        return normalized_batch, chars_count

    def predict_batch(self, batch_images, long_info=True):
        """
        batch预测
        :param batch_images:
        :return:
        """
        start_time = time.time()
        batch_texts = []
        batch_images_idx = []
        invalid_images_idx = []
        image_widths = []
        image_heights = []
        for i, image in enumerate(batch_images):
            image = image_to_pil(image)
            image_widths.append(image.width)
            image_heights.append(image.height)
            image = self.normalize_image(image)
            if image.width <= 4:
                invalid_images_idx.append(i)
                batch_images_idx.append(i)
                batch_images[i] = Image.new('L', (32, 32), color=255)
                image_widths[i] = 32
                image_heights[i] = 32
                continue
            batch_images[i] = image
            batch_images_idx.append(i)

        images_with_idx = zip(batch_images, image_widths, image_heights, batch_images_idx)
        batch_images, image_widths, image_heights, batch_images_idx = zip(
            *sorted(images_with_idx, key=lambda x: x[0].width))
        rets = []

        number_images = len(batch_images)
        number_batches = number_images // BATCH_SIZE
        number_remained = number_images % BATCH_SIZE
        if number_remained == 0:
            iters = number_batches
        else:
            iters = number_batches + 1

        for step in range(iters):
            offset = step * BATCH_SIZE
            batch_array = batch_images[offset:min(offset + BATCH_SIZE, number_images)]
            batch_array, chars_count = self.normalize_batch(batch_array)

            feed = {self.net.feeds()['inputs']: batch_array,
                    self.net.feeds()['is_training']: False
                    }
            predict_label, predict_prob, logits, cnn_out = self.session.run(self.net.fetches(), feed_dict=feed)

            if number_remained > 0 and step == number_batches:
                predict_label = predict_label[:number_remained]
                predict_prob = predict_prob[:number_remained]
                chars_count = chars_count[:number_remained]
            for i in range(len(predict_label)):
                width = image_widths[step * BATCH_SIZE + i]
                height = image_heights[step * BATCH_SIZE + i]
                count = int(chars_count[i])
                label = predict_label[i][:count]
                prob = predict_prob[i][:count]
                p, weights, positions = ctc_label(label, prob, width,
                                                  blank_index=self.label_converter.num_classes - 1)
                txt = self.label_converter.decode(p, invalid_index=-1)
                ret = dict()
                ret['label'] = label
                ret['text'] = txt
                ret['weights'] = [float(weight) for weight in weights[:len(txt)]]
                ret['positions'] = [float(position) for position in positions[:len(txt)]]
                ret['direction'] = 0

                if ret['text'] != '':
                    ret['score'] = float(np.mean(ret['weights']))
                else:
                    ret['score'] = 0

                rets.append(ret)

        for i in range(len(batch_images)):
            ret = rets[batch_images_idx.index(i)]
            batch_texts.append(ret)
        for i in invalid_images_idx:
            ret = dict()
            ret['label'] = [0]
            ret['weights'] = [0]
            ret['positions'] = [0]
            ret['direction'] = 0
            ret['score'] = 0
            ret['text'] = ''
            batch_texts[i] = ret
        print('predict_batch time is %.4f ms' % ((time.time() - start_time) * 1000))
        if long_info:
            return [json.dumps(text, ensure_ascii=False) for text in batch_texts]
        else:
            return [ret['text'] for ret in batch_texts]
