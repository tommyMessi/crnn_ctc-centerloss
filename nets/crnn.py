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
        # single char labels required by center_loss op
        self.bat_labels = tf.placeholder(tf.int32, shape=[None], name="bat_labels")
        # sequence length
        self.len_labels = tf.placeholder(tf.int32, name="len_labels")
        # nums of chars in each sample, used to filter sample to do center loss
        self.char_num = tf.placeholder(tf.int32, shape=[None], name="char_num")
        # char pos: the positions of chars
        # 因为 tensorflow 对在循环中长度改变的张量会报错，所以在此作为 placeholder 传入
        self.char_pos_init = tf.placeholder(tf.int32, shape=[None, 2], name='char_pos')
        # 1d array of size [batch_size]
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.num_classes = num_classes

        self._build_model()
        self._build_train_op()

        self.merged_summary = tf.summary.merge_all()

    @staticmethod
    def pr_shape(tensor):
        return tf.Print(tensor, [tf.shape(tensor)], tensor.name, summarize=100, name='print_shape')

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
                        # 获取全连接之前的数组，即字符图像的 embedding
                        bilstm, embedding = self._bidirectional_LSTM(bilstm, self.num_classes)
                    else:
                        bilstm, _ = self._bidirectional_LSTM(bilstm, self.cfg.rnn_num_units)
            logits = bilstm
        else:
            logits = slim.fully_connected(cnn_out_reshaped, self.num_classes, activation_fn=None)
            embedding = cnn_out_reshaped

        self.embedding = embedding
        self.prelogits = tf.reshape(logits, [-1, self.num_classes])
        self.logits = tf.transpose(logits, (1, 0, 2))

        self.raw_pred = tf.argmax(logits, axis=2, name='raw_prediction')
        raw_prob = tf.nn.softmax(logits)
        top2_probs, top2_pred = tf.nn.top_k(raw_prob, k=2, sorted=True, name="top2")
        self.top2 = top2_pred
        self.probs = raw_prob

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

        # 使用单个样本的对齐策略，如果一个样本中有重复预测，则去重后参与 center_loss 计算，如果有漏字，则不参与 center_loss 计算
        # 生成参与 center loss 计算的 embedding features 和标签
        self.raw_pred_to_features(self.raw_pred, self.bat_labels, self.embedding,
                                  self.char_num, self.char_pos_init)

        # 计算 center loss
        self.center_loss, centers, self.centers_update_op = self.get_center_loss(self.embedding,
                                                                                 self.char_label, 0.05, 6941, True)

        self.total_loss = self.ctc_loss + self.center_loss*0.00001

        tf.summary.scalar('ctc_loss', self.ctc_loss)
        tf.summary.scalar('center_loss', self.center_loss)
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
        # update_ops.append(centers_update_op)
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
        lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(self._LSTM_cell(),
                                                     self._LSTM_cell(),
                                                     inputs,
                                                     sequence_length=self.seq_len,
                                                     dtype=tf.float32)

        lstm_out = tf.concat(lstm_out, 2)
        outputs = tf.reshape(lstm_out, [-1, self.cfg.rnn_num_units * 2])

        outputs = slim.fully_connected(outputs, num_out, activation_fn=None)

        shape = tf.shape(inputs)
        outputs = tf.reshape(outputs, [shape[0], -1, num_out])

        # 为获取字符的 embedding，需要 lstm_out 数组
        return outputs, lstm_out

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
                'len_labels':self.len_labels,
                'is_training': self.is_training}

    def get_center_loss(self, features, labels, alpha, num_classes, verbose=False):
        """获取center loss及center的更新op

        Arguments:
            features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
            labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
            alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
            num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
            verbose: 打印中间过程

        Return：
            loss: Tensor,可与softmax loss相加作为总的loss进行优化.
            centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
            centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        """
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]

        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])
        print('tf.shape(labels):', tf.shape(labels))

        # 构建label
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

        # 打印各种距离变化的过程。加上 verbose 有利于观察变化过程，但可能训练变慢、占用空间变大
        self.max_k, self.min_k = None, None
        if verbose:
            # 获取距离最近的字符，及所谓"形近字"
            labels_brother, label_prob = self.get_nearest()
            centers_brother = tf.gather(centers, labels_brother)

            # 计算形近字类别中心之间的距离
            center_distance = tf.reduce_mean(tf.norm(centers_batch - centers_brother, axis=1))
            self.center_distance = tf.Print(center_distance, [center_distance], 'centers之间的距离')
            tf.summary.scalar(name='dist_between_centers',
                              tensor=self.center_distance)

            # 计算字符到形近字中心的距离
            to_brother_center = tf.norm(features - centers_brother, axis=1)
            distance_to_brother = tf.reduce_mean(to_brother_center)
            self.distance_to_brother = tf.Print(distance_to_brother, [distance_to_brother], '与形近字center的平均距离')
            tf.summary.scalar(name='dist_to_brother_center',
                              tensor=self.distance_to_brother)

            # 计算字符到自己类别中心的距离
            to_self_center = tf.norm(features - centers_batch, axis=1)
            distance_to_self = tf.reduce_mean(to_self_center)
            self.distance_to_self = tf.Print(distance_to_self, [distance_to_brother], '与自己center的平均距离')
            tf.summary.scalar(name='dist_to_self_center',
                              tensor=self.distance_to_self)

            # 计算距离最大与最小的字符，距离指的是 字符距自身中心的距离 - 字符距形近字中心的距离
            diff = to_brother_center - to_self_center

            values, indices = tf.nn.top_k(diff, k=3)
            max_k = tf.gather(labels, indices)
            max_probs = tf.gather(label_prob, indices)
            self.max_k = max_k, values, max_probs

            values, indices = tf.nn.top_k(- diff, k=3)
            min_k = tf.gather(labels, indices)
            min_probs = tf.gather(label_prob, indices)
            self.min_k = min_k, - values, min_probs

        return loss, centers, centers_update_op


    @tf.function
    def get_char_pos_and_label(self, preds, label, char_num, poses):
        """
        过滤掉预测漏字的样本，返回过滤后的字符位置和标签
        Args:
            preds: 去掉重复字符后的预测结果，是字符的位置为 True，否则为 False
            label: 字符标签
            char_num: 每个样本的字符数
            poses: 初始化的字符位置

        Returns:
            字符位置: 2D tensor of shape (num of chars, 2)，后一个维度为（字符位置，图片序号）
            标签：1D tensor of shape (num of chars,)

        """
        i = tf.constant(0, dtype=tf.int32)
        char_total = tf.constant(0, dtype=tf.int32)

        for char in preds:
            char_pos = tf.cast(tf.where(char), tf.int32)

            # 判断预测出的字符数和 gt 是否一致，如果不一致则忽略此样本
            char_seg_num = tf.shape(char_pos)[0]
            if self.is_training:
                if not tf.equal(char_seg_num, char_num[i]):
                    tf.print('切出的字符数量与真实值不同，忽略此样本：',
                             label[char_total:char_total + char_num[i]], char_seg_num, 'vs', char_num[i], summarize=-1)
                    label = tf.concat([label[:char_total], label[char_total + char_num[i]:]], axis=0)
                    i = tf.add(i, 1)
                    continue
                else:
                    char_total = tf.add(char_total, char_num[i])

            # 在seg中添加 batch 序号标识，方便后续获取 feature
            batch_i = char_pos[:, :1]
            batch_i = tf.broadcast_to(i, tf.shape(batch_i))
            char_pos = tf.concat([char_pos, batch_i], axis=1, name='add_batch_index')

            # 连接在一个 segs tensor 上
            poses = tf.concat([poses, char_pos], axis=0, name='push_in_segs')
            i = tf.add(i, 1)

        return poses[1:], label


    @staticmethod
    def get_features(char_pos, embedding):
        """
        根据字符的位置从相应时间步中获取 features
        Args:
            char_pos: 字符位置，2D tensor of shape (num of chars, 2)，最后一个维度为（字符位置，图片序号）
            embedding: 输入全连接层的 tensor

        Returns:
            features: 字符对应的 feature

        """

        def get_slice(pos):
            feature_one_char = embedding[pos[1], pos[0], :]
            return feature_one_char

        features = tf.map_fn(get_slice, char_pos, dtype=tf.float32)

        return features

    @tf.function
    def get_near(self, _input):
        """

        Args:
            _input:

        Returns:

        """

        pos, label = _input
        pred = self.top2[pos[1], pos[0], 0]
        prob = self.probs[pos[1], pos[0], label]
        if tf.equal(pred, label):
            pred = self.top2[pos[1], pos[0], 1]
        return pred, prob

    def get_nearest(self):
        """
        找到和真实值不同的、预测概率最大的字符，即所谓形近字

        Returns:
            nearest: 预测结果对应的形近字
            gt_prob: 正确字符的预测置信度

        """
        stuff = tf.map_fn(self.get_near, (self.char_pos, self.char_label), dtype=(tf.int32, tf.float32))
        nearest, gt_prob = stuff

        return nearest, gt_prob

    def raw_pred_to_features(self, raw_pred, label, embedding, char_num, poses):
        """
        得到用于计算 centerloss 的 embedding features，和对应的标签
        Args:
            raw_pred: 原始的预测结果，形如 [[6941, 6941, 0, 6941, 6941, 5, 6941], …]
            label: 字符标签，形如 [0,5,102,10,…]
            embedding: 全连接的输入张量
            char_num: 每个样本的字符数，用于校验是否可以对齐
            poses: 初始化的字符位置

        Returns:
            self.embedding: embedding features
            self.char_label: 和 embedding features 对应的标签
            self.char_pos: 和 embedding features 对应的字符位置

        """
        with tf.variable_scope('pos'):
            # 判断是否为预测的字符
            is_char = tf.less(raw_pred, self.num_classes - 1)

            # 错位比较法，找到重复字符
            char_rep = tf.equal(raw_pred[:, :-1], raw_pred[:, 1:])
            tail = tf.greater(raw_pred[:, :1], self.num_classes - 1)
            char_rep = tf.concat([char_rep, tail], axis=1)

            # 去掉重复字符之后的字符位置，重复字符取其 最后一次 出现的位置
            char_no_rep = tf.math.logical_and(is_char, tf.math.logical_not(char_rep))

            # 得到字符位置 和 相应的标签，如果某张图片 预测出来的字符数量 和gt不一致则跳过
            self.char_pos, self.char_label = self.get_char_pos_and_label(preds=char_no_rep,
                                                                         label=label,
                                                                         char_num=char_num,
                                                                         poses=poses)
            # 根据字符位置得到字符的 embedding
            self.embedding = self.get_features(self.char_pos, embedding)


if __name__ == '__main__':
    from libs.label_converter import LabelConverter

    cfg = load_config('raw')
    converter = LabelConverter(chars_file='./data/chars/lexicon.txt')
    model = CRNN(cfg, num_classes=converter.num_classes)