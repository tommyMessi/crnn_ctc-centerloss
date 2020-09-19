"""
Modified from https://github.com/deepinsight/insightocr/blob/master/crnn/symbols/simplenet.py
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils


class SimpleNet(object):
    def __init__(self, inputs, is_training):
        self._scope = 'simple_net'

        with slim.arg_scope(self._arg_scope(is_training)):
            self.build_net(inputs, is_training)

    def build_net(self, inputs, is_training):
        """
        feature_maps = [64, 128, 256, 512, 512, 512]
        """
        with tf.variable_scope(self._scope, self._scope, [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.batch_norm],
                                outputs_collections=end_points_collection):
                net = slim.conv2d(inputs, 64, 3, 1, scope='conv1')

                max_pool = slim.max_pool2d(net, 2, 2, scope='max_pool1')
                avg_pool = slim.avg_pool2d(net, 2, 2, scope='avg_pool1')

                net = max_pool - avg_pool

                net = slim.conv2d(net, 128, 3, 1, scope='conv2')
                net = slim.max_pool2d(net, 2, 2, scope='max_pool2')
                net = slim.conv2d(net, 256, 3, scope='conv3')
                net = slim.conv2d(net, 512, 3, scope='conv5')
                net = slim.max_pool2d(net, 2, [2, 2], scope='max_pool4')
                net = slim.conv2d(net, 512, 3, scope='conv6')
                net = slim.conv2d(net, 512, 2, scope='conv7')

                net = slim.avg_pool2d(net, (4, 1), stride=1)
                net = slim.dropout(net, keep_prob=0.5)

            self.end_points = utils.convert_collection_to_dict(end_points_collection)
            self.net = net

    def _arg_scope(self, is_training):
        batch_norm_params = {
            'is_training': is_training,
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.00001
        }

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.leaky_relu,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.dropout], is_training=is_training) as sc:
                return sc


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '../../libs')
    from tf_utils import print_endpoints

    inputs = tf.placeholder(tf.float32, [1, 32, 256, 1], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_training")
    img_file = '/home/cwq/data/ocr/train_data/400w_eng_corpus/val/00000000.jpg'

    net = SimpleNet(inputs, is_training)
    print_endpoints(net, inputs, is_training, img_file)
