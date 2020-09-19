import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.cnn.mobilenet import conv_blocks as ops
from nets.cnn.mobilenet import mobilenet as lib
import nets.cnn.mobilenet.mobilenet_v2 as mobilenet_v2

op = lib.op
expand_input = ops.expand_input_by_factor

V2_CRNN = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=1, num_outputs=8, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(slim.max_pool2d, stride=(2, 1), kernel_size=2),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=2, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=160),
        op(ops.expanded_conv, stride=1, num_outputs=320),
        op(slim.avg_pool2d, stride=1, kernel_size=(4, 1)),
    ],
)


class MobileNetV2(object):
    def __init__(self, inputs, is_training):
        with slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
            logits, endpoints = mobilenet_v2.mobilenet_base(inputs, conv_defs=V2_CRNN)

        self.net = logits
        self.end_points = endpoints


if __name__ == '__main__':
    import sys

    sys.path.insert(0, '../../libs')
    from tf_utils import print_endpoints

    inputs = tf.placeholder(tf.float32, [1, 32, 256, 1], name="inputs")
    is_training = tf.placeholder(tf.bool, name="is_training")
    img_file = '/home/cwq/data/ocr/train_data/400w_eng_corpus/val/00000000.jpg'

    res_net = MobileNetV2(inputs, is_training)
    print_endpoints(res_net, inputs, is_training, img_file)
