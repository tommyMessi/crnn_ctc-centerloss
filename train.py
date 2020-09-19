import os
import time
import math

from libs.config import load_config

RNG_SEED = 42
import numpy as np

np.random.seed(RNG_SEED)
import tensorflow as tf
import cv2
tf.set_random_seed(RNG_SEED)

import libs.utils as utils
import libs.tf_utils as tf_utils
from libs.img_dataset import ImgDataset
from libs.label_converter import LabelConverter
import libs.infer as infer

from nets.crnn import CRNN
from _pydecimal import Decimal, Context, ROUND_HALF_UP

class Trainer(object):
    def __init__(self):
        pass


    def train(self, log_dir, restore, log_step, ckpt_dir, val_step, cfg_name, chars_file, train_txt, val_txt, test_txt, result_dir):

        cfg = load_config(cfg_name)

        converter = LabelConverter(chars_file=chars_file)

        tr_ds = ImgDataset(train_txt, converter, cfg.batch_size)

        cfg.lr_boundaries = [10000]
        cfg.lr_values = [cfg.lr * (cfg.lr_decay_rate ** i) for i in
                              range(len(cfg.lr_boundaries) + 1)]

        if val_txt is None:
            val_ds = None
        else:
            val_ds = ImgDataset(val_txt, converter, cfg.batch_size, shuffle=False)

        if test_txt is None:
            test_ds = None
        else:
            # Test images often have different size, so set batch_size to 1
            test_ds = ImgDataset(test_txt, converter, shuffle=False, batch_size=1)

        model = CRNN(cfg, num_classes=converter.num_classes)

        epoch_start_index = 0
        batch_start_index = 0


        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=8)
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        if restore:
            self._restore(sess, saver, model,tr_ds, ckpt_dir)

        print('Begin training...')
        for epoch in range(epoch_start_index, cfg.epochs):
            sess.run(tr_ds.init_op)

            for batch in range(batch_start_index, tr_ds.num_batches):
                batch_start_time = time.time()

                if batch != 0 and (batch %  log_step == 0):
                    batch_cost, global_step, lr = self._train_with_summary( model, tr_ds, sess, train_writer, converter)
                else:
                    batch_cost, global_step, lr = self._train(model, tr_ds, sess)

                print("epoch: {}, batch: {}/{}, step: {}, time: {:.02f}s, loss: {:.05}, lr: {:.05}"
                      .format(epoch, batch, tr_ds.num_batches, global_step, time.time() - batch_start_time,
                              batch_cost, lr))

                if global_step != 0 and (global_step % val_step == 0):
                    val_acc = self._do_val(val_ds, epoch, global_step, "val", sess, model, converter,  train_writer, cfg, result_dir)
                    test_acc = self._do_val(test_ds, epoch, global_step, "test", sess, model, converter, train_writer, cfg, result_dir)
                    self._save_checkpoint(ckpt_dir, global_step, saver, sess, val_acc, test_acc)

            batch_start_index = 0

    def _restore(self, sess, saver,  model, tr_ds, ckpt_dir):
        utils.restore_ckpt(sess, saver, ckpt_dir)

        step_restored = sess.run(model.global_step)

        epoch_start_index = math.floor(step_restored / tr_ds.num_batches)
        batch_start_index = step_restored % tr_ds.num_batches

        print("Restored global step: %d" % step_restored)
        print("Restored epoch: %d" % epoch_start_index)
        print("Restored batch_start_index: %d" % batch_start_index)

    def round_up(self, n):
        print(n * 10 % 10)
        k = n * 10 % 10
        if k < 5:
            return int(n)
        else:
            return int(n) + 1

    def _train(self, model, tr_ds, sess):
        img_batch, label_batch, labels, positions, _ = tr_ds.get_next_batch(sess)

        image_batch_shape = img_batch.shape
        w = self.round_up(image_batch_shape[2]/4)
        print(w)
        positions_list = []
        for position_str in positions:
            list2 = [6940 for i in range(w)]
            position_list = str(position_str, encoding = "utf8").split(',')
            num_list_new = [int(x) for x in position_list]
            list2[:len(num_list_new)] = num_list_new

            positions_list.append(list2)
        con_labels_batch = np.array(positions_list)
        con_labels_batch = con_labels_batch.reshape((-1))


        feed = {model.inputs: img_batch,
                model.labels: label_batch,
                model.con_labels: con_labels_batch,
                model.len_labels: w,
                model.is_training: True}

        fetches = [model.total_loss,
                   model.ctc_loss,
                   model.outputs_center,
                   model.regularization_loss,
                   model.global_step,
                   model.lr,
                   model.train_op]

        batch_cost, ctc_loss ,centers_update_op, _, global_step, lr, _ = sess.run(fetches, feed)
        print('center_loss',centers_update_op.shape)
        print('ctc_loss', ctc_loss)
        return batch_cost, global_step, lr

    def _train_with_summary(self, model, tr_ds, sess, train_writer,converter):
        img_batch, label_batch, labels, positions, _ = tr_ds.get_next_batch(sess)
        image_batch_shape = img_batch.shape
        w = self.round_up(image_batch_shape[2]/4)
        positions_list = []
        for position_str in positions:
            list2 = [6940 for i in range(w)]
            position_list = str(position_str, encoding = "utf8").split(',')
            num_list_new = [int(x) for x in position_list]
            list2[:len(num_list_new)] = num_list_new

            positions_list.append(list2)
        con_labels_batch = np.array(positions_list)
        con_labels_batch = con_labels_batch.reshape((-1))

        # print('image_batch:',img_batch)
        feed = {model.inputs: img_batch,
                model.labels: label_batch,
                model.con_labels: con_labels_batch,
                model.len_labels: w,
                model.is_training: True}

        fetches = [model.total_loss,
                   model.centers_update_op,
                   model.ctc_loss,
                   model.regularization_loss,
                   model.global_step,
                   model.lr,
                   model.merged_summay,
                   model.dense_decoded,
                   model.edit_distance,
                   model.train_op]

        batch_cost, centers_update_op,_, _, global_step, lr, summary, predicts, edit_distance, _ = sess.run(fetches, feed)
        train_writer.add_summary(summary, global_step)

        predicts = [converter.decode(p, CRNN.CTC_INVALID_INDEX) for p in predicts]
        accuracy, _ = infer.calculate_accuracy(predicts, labels)

        tf_utils.add_scalar_summary(train_writer, "train_accuracy", accuracy, global_step)
        tf_utils.add_scalar_summary(train_writer, "train_edit_distance", edit_distance, global_step)

        return batch_cost, global_step, lr

    def _do_val(self, dataset, epoch, step, name, sess, model, converter, train_writer, cfg, result_dir):
        if dataset is None:
            return None

        accuracy, edit_distance = infer.validation(sess, model.feeds(), model.fetches(),
                                                   dataset, converter, result_dir, name, step)

        tf_utils.add_scalar_summary(train_writer, "%s_accuracy" % name, accuracy, step)
        tf_utils.add_scalar_summary(train_writer, "%s_edit_distance" % name, edit_distance, step)

        print("epoch: %d/%d, %s accuracy = %.3f" % (epoch, cfg.epochs, name, accuracy))
        return accuracy

    def _save_checkpoint(self, ckpt_dir, step, saver, sess, val_acc=None, test_acc=None):
        ckpt_name = "crnn_%d" % step
        if val_acc is not None:
            ckpt_name += '_val_%.03f' % val_acc
        if test_acc is not None:
            ckpt_name += '_test_%.03f' % test_acc

        name = os.path.join(ckpt_dir, ckpt_name)
        print("save checkpoint %s" % name)

        meta_exists, meta_file_name = self._meta_file_exist(ckpt_dir)

        saver.save(sess, name)

        # remove old meta file to save disk space
        if meta_exists:
            try:
                os.remove(os.path.join(ckpt_dir, meta_file_name))
            except:
                print('Remove meta file failed: %s' % meta_file_name)

    def _meta_file_exist(self, ckpt_dir):
        fnames = os.listdir(ckpt_dir)
        meta_exists = False
        meta_file_name = ''
        for n in fnames:
            if 'meta' in n:
                meta_exists = True
                meta_file_name = n
                break

        return meta_exists, meta_file_name


def main():
    dev = '/gpu:0'
    with tf.device(dev):
        trainer = Trainer()
        trainer.train(log_dir='./output_20200918/log', restore=False, log_step=50, val_step=50, cfg_name='raw', ckpt_dir='./output_20200918/checkpoint/default',
                      chars_file='./data/chars/lexicon.txt', train_txt='./data_example/train_new.txt',
                      val_txt='./data_example//test_new.txt', test_txt='./data_example//test_new.txt',
                      result_dir='./output_20200918/result')


if __name__ == '__main__':
    main()
