import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils


class PaperCNN(object):
    def __init__(self, inputs, is_training):
        self._scope = 'crnn_cnn'

        self.build_net(inputs, is_training)

    def build_net(self, inputs, is_training):
        """
        Net structure described in crnn paper
        feature_maps = [64, 128, 256, 256, 512, 512, 512]
        """
        norm_params = {
            'is_training': is_training,
            'decay': 0.9,
            'epsilon': 1e-05
        }

        with tf.variable_scope(self._scope, self._scope, [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.batch_norm],
                                outputs_collections=end_points_collection):
                net = slim.conv2d(inputs, 64, 3, 1, scope='conv1')
                net = slim.max_pool2d(net, 2, 2, scope='pool1')
                net = slim.conv2d(net, 128, 3, 1, scope='conv2')
                net = slim.max_pool2d(net, 2, 2, scope='pool2')
                net = slim.conv2d(net, 256, 3, scope='conv3')
                net = slim.conv2d(net, 256, 3, scope='conv4')
                net = slim.max_pool2d(net, 2, [2, 1], scope='pool3')
                net = slim.conv2d(net, 512, 3, normalizer_fn=slim.batch_norm, normalizer_params=norm_params,
                                  scope='conv5')
                net = slim.conv2d(net, 512, 3, normalizer_fn=slim.batch_norm, normalizer_params=norm_params,
                                  scope='conv6')
                net = slim.max_pool2d(net, 2, [2, 1], scope='pool4')
                net = slim.conv2d(net, 512, 2, padding='VALID', scope='conv7')

            self.end_points = utils.convert_collection_to_dict(end_points_collection)
            self.net = net


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '../../libs')
    from tf_utils import print_endpoints

    inputs = tf.placeholder(tf.float32, [1, 32, 256, 1], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_training")
    img_file = '/home/cwq/data/ocr/train_data/400w_eng_corpus/val/00000000.jpg'

    net = PaperCNN(inputs, is_training)
    print_endpoints(net, inputs, is_training, img_file)
