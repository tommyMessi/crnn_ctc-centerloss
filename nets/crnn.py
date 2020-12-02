import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.config import load_config
from nets.cnn.mobile_net_v2 import MobileNetV2
from nets.cnn.paper_cnn import PaperCNN
from nets.cnn.dense_net import DenseNet
from nets.cnn.squeeze_net import SqueezeNet
from nets.cnn.resnet_v2 import ResNetV2
from nets.cnn.simple_net import SimpleNet
import numpy as np
from PIL import Image
import copy

class CRNN(object):
    CTC_INVALID_INDEX = -1

    def __init__(self, cfg, num_classes):
        self.inputs = tf.placeholder(tf.float32,
                                     [None, 32, None, 1],
                                     name="inputs")
        self.cfg = cfg
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32, name="labels")
        self.bat_labels = tf.placeholder(tf.int32, name="bat_labels")
        self.con_labels = tf.placeholder(tf.int32, name="con_labels")
        self.len_labels = tf.placeholder(tf.int32, name="len_labels")
        # 1d array of size [batch_size]
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.num_classes = num_classes

        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        if self.cfg.name == 'raw':
            net = PaperCNN(self.inputs, self.is_training)
        elif self.cfg.name == 'dense':
            net = DenseNet(self.inputs, self.is_training)
        elif self.cfg.name == 'squeeze':
            net = SqueezeNet(self.inputs, self.is_training)
        elif self.cfg.name == 'resnet':
            net = ResNetV2(self.inputs, self.is_training)
        elif self.cfg.name == 'simple':
            net = SimpleNet(self.inputs, self.is_training)
        elif self.cfg.name == 'mobile':
            net = MobileNetV2(self.inputs, self.is_training)

        # tf.reshape() vs Tensor.set_shape(): https://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape
        # tf.shape() vs Tensor.get_shape(): https://stackoverflow.com/questions/37096225/how-to-understand-static-shape-and-dynamic-shape-in-tensorflow
        cnn_out = net.net
        self.cnn_out = cnn_out
        cnn_output_shape = tf.shape(cnn_out)# 32 , 4, 64, 1024
        print('tf.shape(cnn_out):',tf.shape(cnn_out))
        batch_size = cnn_output_shape[0]
        self.batch_size = batch_size
        cnn_output_h = cnn_output_shape[1]
        cnn_output_w = cnn_output_shape[2]
        cnn_output_channel = cnn_output_shape[3]

        # Get seq_len according to cnn output, so we don't need to input this as a placeholder
        self.seq_len = tf.ones([batch_size], tf.int32) * cnn_output_w

        # Reshape to the shape lstm needed. [batch_size, max_time, ..]
        cnn_out_transposed = tf.transpose(cnn_out, [0, 2, 1, 3])
        cnn_out_reshaped = tf.reshape(cnn_out_transposed, [batch_size, cnn_output_w, cnn_output_h * cnn_output_channel])

        cnn_shape = cnn_out.get_shape().as_list()
        cnn_out_reshaped.set_shape([None, cnn_shape[2], cnn_shape[1] * cnn_shape[3]])

        if self.cfg.use_lstm:
            bilstm = cnn_out_reshaped
            for i in range(self.cfg.num_lstm_layer):
                with tf.variable_scope('bilstm_%d' % (i + 1)):
                    if i == (self.cfg.num_lstm_layer - 1):
                        bilstm = self._bidirectional_LSTM(bilstm, self.num_classes)
                    else:
                        bilstm = self._bidirectional_LSTM(bilstm, self.cfg.rnn_num_units)
            logits = bilstm
        else:
            logits = slim.fully_connected(cnn_out_reshaped, self.num_classes, activation_fn=None)


        self.prelogits = tf.reshape(logits, [-1, self.num_classes])
        self.logits = tf.transpose(logits, (1, 0, 2))
        self.outputs_center = tf.reshape(self.logits[:, :self.len_labels, :], [-1, self.num_classes])

    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)


        self.ctc_loss = tf.nn.ctc_loss(labels=self.labels,
                                       inputs=self.logits,
                                       ignore_longer_outputs_than_inputs=True,
                                       sequence_length=self.seq_len)
        # self.pro_loss = tf.reduce_mean(self.predict_prob)
        self.ctc_loss = tf.reduce_mean(self.ctc_loss)
        self.regularization_loss = tf.constant(0.0)
        # self.regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        max_array =  tf.argmax(self.logits, axis=2)

        self.ind_array = tf.where(condition=max_array<6940)

        center_max_array = tf.argmax(self.outputs_center, axis=1)
        self.center_ind_array = tf.where(condition=center_max_array<6940)

        self.center_input_tensor = tf.gather(self.outputs_center, self.center_ind_array, axis=0)
        self.center_input_tensor = tf.squeeze(self.center_input_tensor)

        self.center_loss, centers, self.centers_update_op = self.get_center_loss(self.center_input_tensor, self.bat_labels, 0.5, 6941)

        self.total_loss = self.ctc_loss + self.center_loss*0.000001


        tf.summary.scalar('ctc_loss', self.ctc_loss)
        tf.summary.scalar('regularization_loss', self.regularization_loss)
        tf.summary.scalar('total_loss', self.total_loss)


        self.lr = tf.train.piecewise_constant(self.global_step, self.cfg.lr_boundaries, self.cfg.lr_values)

        tf.summary.scalar("learning_rate", self.lr)

        if self.cfg.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.cfg.optimizer == 'rms':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                       epsilon=1e-8)
        elif self.cfg.optimizer == 'adadelate':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr,
                                                        rho=0.9,
                                                        epsilon=1e-06)
        elif self.cfg.optimizer == 'sgd':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,
                                                        momentum=0.9)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if self.centers_update_op!=0:
            update_ops.append(self.centers_update_op)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

        # inputs shape: [max_time x batch_size x num_classes]

        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len, merge_repeated=True)

        print('decoded',self.decoded)
        # dense_decoded shape: [batch_size, encoded_code_size(not fix)]
        # use tf.cast here to support run model on Android
        self.dense_decoded = tf.sparse_tensor_to_dense(tf.cast(self.decoded[0], tf.int32),
                                                       default_value=self.CTC_INVALID_INDEX, name="output")

        # Edit distance for wrong result
        self.edit_distances = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels)

        non_zero_indices = tf.where(tf.not_equal(self.edit_distances, 0))
        self.edit_distance = tf.reduce_mean(tf.gather(self.edit_distances, non_zero_indices))

    def _LSTM_cell(self, num_proj=None):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.cfg.rnn_num_units, num_proj=num_proj)
        if self.cfg.rnn_keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.cfg.rnn_keep_prob)
        return cell

    def _paper_bidirectional_LSTM(self, inputs, num_proj):
        """
            根据 CRNN BiRnnJoin.lua 源码改写
        :param inputs: shape [batch_size, max_time, ...]
        :param num_proj: 每个 cell 输出的维度
        :return: shape [batch_size, max_time, num_proj]
        """
        (blstm_fw, blstm_bw), _ = tf.nn.bidirectional_dynamic_rnn(self._LSTM_cell(num_proj=num_proj),
                                                                  self._LSTM_cell(num_proj=num_proj),
                                                                  inputs,
                                                                  sequence_length=self.seq_len,
                                                                  dtype=tf.float32)
        return tf.add(blstm_fw, blstm_bw)

    def _bidirectional_LSTM(self, inputs, num_out):
        #numout == 6941
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self._LSTM_cell(),
                                                     self._LSTM_cell(),
                                                     inputs,
                                                     sequence_length=self.seq_len,
                                                     dtype=tf.float32)

        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, self.cfg.rnn_num_units * 2])

        outputs = slim.fully_connected(outputs, num_out, activation_fn=None)

        shape = tf.shape(inputs)
        outputs = tf.reshape(outputs, [shape[0], -1, num_out])

        return outputs

    def fetches(self):
        """
        Return operations to fetch for inference
        """
        return [
            self.log_prob,
            self.dense_decoded,
            self.edit_distance,
            self.edit_distances,

        ]


    def feeds(self):
        """
        Return placeholders to feed for inference
        """
        return {'inputs': self.inputs,
                'labels': self.labels,
                'con_labels':self.con_labels,
                'len_labels':self.len_labels,
                'is_training': self.is_training}

    def get_center_loss(self, features, labels, alpha, num_classes):
        """获取center loss及center的更新op

        Arguments:
            features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
            labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
            alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
            num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

        Return：
            loss: Tensor,可与softmax loss相加作为总的loss进行优化.
            centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
            centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        """
        # 获取特征的维数，例如256维
        print('~~~~~~~~~~~~~~~~~~~~~~~~',labels.get_shape())
        print('~~~~~~~~~~~~~~~~~~~~~~~~',features.get_shape())
        len_features = features.get_shape()[1]

        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        print('~~~~~~~~~~~~~~~~~~~~~~~',num_classes, len_features)
        centers = tf.get_variable('centers', [num_classes, 6941], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])
        print('tf.shape(labels):', tf.shape(labels))

        # 构建label

        if tf.shape(centers) != tf.shape(labels):
            return 0,0,0
        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss = tf.nn.l2_loss(features - centers_batch)

        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)

        return loss, centers, centers_update_op

