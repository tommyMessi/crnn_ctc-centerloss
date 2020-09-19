# coding:utf-8
from __future__ import unicode_literals

import os
import cv2
import json
import copy
import time
import codecs
import logging

import numpy as np
from PIL import Image
from infer import Infer

BATCH_SIZE = 32


def test_ocr_model(img_dir, gt_file, report_file):
    with codecs.open(gt_file, 'r', encoding='utf-8') as f:
        test_texts = f.readlines()

    with codecs.open(report_file, 'w', encoding='utf-8') as f:
        correct_count = 0
        for n in range(len(test_texts)):
            split_index = test_texts[n].find('\t')
            img_name = test_texts[n][:split_index]

            ground_text = test_texts[n][split_index + 1:]
            ground_text = ground_text.strip()
            img_name = os.path.join(img_dir, img_name)
            print(img_name)
            try:
                img = Image.open(img_name)
            except:
                print('image not found', img_name)
            if img.mode != 'L':
                img = img.convert('L')
            predict_text = ocr_engine.predict(img, long_info=True)


            f.write('{}\t{}\t{}'.format(img_name, ground_text, predict_text))
            f.flush()
            if predict_text == ground_text:
                correct_count += 1
                # print('correct')
                f.write('\tcorrect\n')
            else:
                # print('wrong')
                f.write('\twrong\n')
        print(correct_count)
        print(correct_count / float(len(test_texts)))


def test_batch_ocr_model(img_dir, gt_file, report_file):
    with codecs.open(gt_file, 'r', encoding='utf-8') as f:
        test_texts = f.readlines()

    with codecs.open(report_file, 'w', encoding='utf-8') as f:
        correct_count = 0
        img_name_batch = []
        img_batch = []
        gt_batch = []
        for n in range(len(test_texts)):
            split_index = test_texts[n].find('\t')
            img_name = test_texts[n][:split_index]
            ground_text = test_texts[n][split_index + 1:len(test_texts[n]) - 1]
            ground_text = ground_text.strip()
            img_name = os.path.join(img_dir, img_name)
            # print(img_name)
            try:
                img = Image.open(img_name)
            except:
                print('image not found', img_name)
                continue
            if img.mode != 'L':
                img = img.convert('L')
            img_name_batch.append(img_name)
            img_batch.append(img)
            gt_batch.append(ground_text)
            if len(img_batch) == BATCH_SIZE or n == len(test_texts) - 1:
                pd_batch = ocr_engine.predict_batch(img_batch, long_info=False)
                print(pd_batch)
                print(gt_batch)
                for i in range(len(img_batch)):
                    f.write('{}\t{}\t{}'.format(img_name_batch[i], gt_batch[i], pd_batch[i]))
                    if gt_batch[i] == pd_batch[i]:
                        correct_count += 1
                        f.write('\tcorrect\n')
                    else:
                        f.write('\twrong\n')
                img_name_batch = []
                img_batch = []
                gt_batch = []
        print(correct_count)
        print(correct_count / float(len(test_texts)))


def output_ocr_result(image_file):
    image_name_list = []
    if os.path.isdir(image_file):
        image_name_list = os.listdir(image_file)
    result_file_name = image_file + '.txt'
    with codecs.open(result_file_name, 'w', encoding='utf-8') as f:
        for i, image_name in enumerate(image_name_list):
            image_path = os.path.join(image_file, image_name)
            try:
                image = Image.open(image_path)
                predict_text = ocr_engine.predict(image, long_info=False)
            except:
                predict_text = ''
            print(image_path)
            print(predict_text)
            f.write('{}\t{}\n'.format(image_path, predict_text))
            f.flush()


ocr_engine = Infer('/home/huluwa/tf_crnn/model/ctc_center')
if __name__ == "__main__":
    TEST_OCR_MODEL = False
    TEST_BATCH_OCR_MODEL = True
    if TEST_OCR_MODEL:
        root_dir = './data_example/test_data/xingjin'
        gt_file = './data_example/test_data/xingjin1'
        report_file = './testset_result_local.txt'
        start_time = time.time()
        test_ocr_model(root_dir, gt_file, report_file)
        print('total cost time is %.4f ms' % ((time.time() - start_time) * 1000))
        exit()
    if TEST_BATCH_OCR_MODEL:
        root_dir = './data_example/test_data/xingjin'
        gt_file = './data_example/test_data/xingjin1'
        report_file = './testset_result_batch.txt'
        start_time = time.time()
        test_batch_ocr_model(root_dir, gt_file, report_file)
        print('total cost time is %.4f ms' % ((time.time() - start_time) * 1000))
        exit()
