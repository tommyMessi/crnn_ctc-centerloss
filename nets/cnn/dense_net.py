import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
from collections import namedtuple

DensenetParams = namedtuple('DensenetParameters', ['first_output_features',
                                                   'layers_per_block',
                                                   'growth_rate',
                                                   'bc_mode',
                                                   'dropout_keep_prob'
                                                   ])


class DenseNet(object):
    default_params = DensenetParams(
        first_output_features=32,
        layers_per_block=20,
        growth_rate=20,
        bc_mode=True,
        dropout_keep_prob=0.8,
    )

    def __init__(self, inputs, params=None, is_training=True):
        if isinstance(params, DensenetParams):
            self.params = params
        else:
            self.params = DenseNet.default_params

        self._scope = 'densenet'

        self.is_training = is_training
        with slim.arg_scope(self.arg_scope(is_training)):
            self.build_net(inputs)

    def build_net(self, inputs):
        num_channels = self.params.first_output_features

        with tf.variable_scope(self._scope, self._scope, [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'

            with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.fully_connected],
                                outputs_collections=end_points_collection):
                with tf.variable_scope("conv_1"):
                    net = slim.conv2d(inputs, self.params.first_output_features, [3, 3])

                with tf.variable_scope('dense_block_1'):
                    niet, num_channels = self.dense_block(net, 24)

                with tf.variable_scope('transition_1'):
                    # feature map size: 32*256 -> 16*128
                    net, num_channels = self.transition_layer(net, 32)

                with tf.variable_scope('dense_block_2'):
                    net, num_channels = self.dense_block(net, 48)

                with tf.variable_scope('transition_2'):
                    # feature map size: 16*128 -> 8*64
                    net, num_channels = self.transition_layer(net, 64)

                with tf.variable_scope('dense_block_3'):
                    net, num_channels = self.dense_block(net, num_channels)

                with tf.variable_scope('transition_3'):
                    # feature map size: 8*64 -> 4*64
                    net, num_channels = self.transition_layer(net, num_channels, pool_stride=[2, 1])

                # with tf.variable_scope('global_average_pooling'):
                #     # net = slim.fully_connected(net, num_channels)
                #     net = slim.avg_pool2d(net, kernel_size=[8, 2])

                # with tf.variable_scope('transition_3'):
                #     # feature map size: 8*64 -> 4*64
                #     net, num_channels = self.transition_layer(net, num_channels, pool_stride=[2, 2],
                #                                               compression_factor=1)
                #
                # with tf.variable_scope('transition_4'):
                #     # feature map size: 4*64 -> 2*64
                #     net, num_channels = self.transition_layer(net, num_channels, pool_stride=[2, 1],
                #                                               compression_factor=1)
                #
                # with tf.variable_scope('transition_5'):
                #     # feature map size: 2*64 -> 1*64
                #     net, num_channels = self.transition_layer(net, num_channels, pool_stride=[2, 1],
                #                                               compression_factor=1)

                self.end_points = utils.convert_collection_to_dict(end_points_collection)
                self.net = net

    def dense_block(self, inputs, num_channels):
        net = slim.repeat(inputs, self.params.layers_per_block, self.block_inner_layer)
        num_channels += self.params.growth_rate * self.params.layers_per_block
        return net, num_channels

    def transition_layer(self, inputs, num_filter, compression_factor=0.5, pool_stride=[2, 2]):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        num_filter = int(compression_factor * num_filter)
        output = self.composite_function(inputs, num_filter, kernel_size=[1, 1])
        output = self.dropout(output)
        output = slim.avg_pool2d(output, [2, 2], stride=pool_stride)
        return output, num_filter

    def block_inner_layer(self, inputs, scope="block_inner_layer"):
        with tf.variable_scope(scope):
            if self.params.bc_mode:
                bottleneck_out = self.bottleneck(inputs)
                _output = self.composite_function(bottleneck_out, self.params.growth_rate)
            else:
                _output = self.composite_function(inputs, self.params.growth_rate)

            output = tf.concat(axis=3, values=(inputs, _output))
            return output

    def bottleneck(self, inputs):
        with tf.variable_scope("bottleneck"):
            num_channels = self.params.growth_rate * 4
            output = slim.batch_norm(inputs)
            output = tf.nn.relu(output)
            output = slim.conv2d(output, num_channels, [1, 1], padding='VALID', activation_fn=None)
            output = self.dropout(output)
        return output

    def dropout(self, inputs):
        return slim.dropout(inputs, self.params.dropout_keep_prob, is_training=self.is_training)

    def composite_function(self, inputs, num_channels, kernel_size=[3, 3]):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            output = slim.batch_norm(inputs)
            output = tf.nn.relu(output)
            output = slim.conv2d(output, num_channels, kernel_size, activation_fn=None)
            output = self.dropout(output)
        return output

    def arg_scope(self, is_training=True,
                  weight_decay=0.0001,
                  batch_norm_decay=0.997,
                  batch_norm_epsilon=1e-5,
                  batch_norm_scale=True):
        batch_norm_params = {
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '../../libs')
    from tf_utils import print_endpoints

    inputs = tf.placeholder(tf.float32, [None, 32, None, 1], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_training")
    img_file = '/home/cwq/data/ocr/train_data/400w_eng_corpus/val/00000000.jpg'

    dense_net = DenseNet(inputs, is_training)
    print_endpoints(dense_net, inputs, is_training, img_file)
