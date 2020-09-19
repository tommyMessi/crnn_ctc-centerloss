import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils


class SqueezeNet(object):
    def __init__(self, inputs, is_training):
        self.is_training = is_training
        self._scope = 'squeezenet'

        with slim.arg_scope(self._arg_scope(is_training)):
            self.build_net(inputs)

    def build_net(self, inputs):
        with tf.variable_scope(self._scope, self._scope, [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.batch_norm],
                                outputs_collections=end_points_collection):
                net = slim.conv2d(inputs, 96, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool1')
                net = self.fire_module(net, 16, 64, scope='fire2')
                net = self.fire_module(net, 16, 64, scope='fire3')
                net = self.fire_module(net, 32, 128, scope='fire4')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool4')
                net = self.fire_module(net, 32, 128, scope='fire5')
                net = self.fire_module(net, 48, 192, scope='fire6')
                net = self.fire_module(net, 48, 192, scope='fire7')
                net = self.fire_module(net, 64, 256, scope='fire8')
                net = slim.max_pool2d(net, [2, 2], stride=[2, 1], scope='maxpool8')
                net = self.fire_module(net, 64, 256, scope='fire9')

        self.end_points = utils.convert_collection_to_dict(end_points_collection)
        self.net = net

    def fire_module(self, inputs,
                    squeeze_depth,
                    expand_depth,
                    reuse=None,
                    scope=None):
        with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
            net = self.squeeze(inputs, squeeze_depth)
            outputs = self.expand(net, expand_depth)
            return outputs

    def squeeze(self, inputs, num_outputs):
        return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='Conv_squeeze')

    def expand(self, inputs, num_outputs):
        with tf.variable_scope('expand'):
            e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='Conv_1x1')
            e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='Conv_3x3')
        return tf.concat([e1x1, e3x3], 3)

    def _arg_scope(self, is_training):
        weight_decay = 0.0

        batch_norm_params = {
            'is_training': is_training,
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.00001
        }

        with slim.arg_scope([slim.conv2d],
                            weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
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

    squeeze_net = SqueezeNet(inputs, is_training)
    print_endpoints(squeeze_net, inputs, is_training, img_file)

