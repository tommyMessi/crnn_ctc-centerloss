from functools import reduce
import os

import tensorflow as tf


def add_scalar_summary(writer, tag, val, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
    writer.add_summary(summary, step)


def print_endpoints(net, inputs, is_training, img_path, CPU=True):
    cnn_output_shape = tf.shape(net.net)
    cnn_output_h = cnn_output_shape[1]
    cnn_output_w = cnn_output_shape[2]
    cnn_output_channel = cnn_output_shape[3]

    cnn_out = tf.transpose(net.net, [0, 2, 1, 3])
    cnn_out = tf.reshape(cnn_out, [-1, cnn_output_w, cnn_output_h * cnn_output_channel])

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=1)

    conv_count = 0
    if CPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        img = sess.run(img_decoded)
        sess.run(net.net, feed_dict={inputs: [img], is_training: True})

        for k, v in net.end_points.items():
            if 'Conv' in k:
                conv_count += 1
            print("%s: %s" % (k, v.shape))

        cnn_out = sess.run(cnn_out, feed_dict={inputs: [img], is_training: True})

        print('-' * 50)
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
        print("Net FLOP: %.02fM" % (flops.total_float_ops / 1000000))

    def size(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list())

    print("-" * 50)

    n = sum(size(v) for v in tf.trainable_variables())
    print("Tensorflow trainable params: %.02fM (%dK)" % (n / 1000000, n / 1000))
    print("Conv layer count: %d" % conv_count)
    print("Output shape: {}".format(net.net))
    print('Cnn out reshaped for lstm: ')
    print(cnn_out.shape)


if __name__ == '__main__':
    # https://stackoverflow.com/questions/45085938/tensorflow-is-there-a-way-to-measure-flops-for-a-model
    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():
        A = tf.Variable(tf.random_normal([25, 16]))
        B = tf.Variable(tf.random_normal([16, 9]))
        C = tf.matmul(A, B)  # shape=[25,9]

        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        # if flops is not None:
        #     print('Flops should be ~', 2 * 25 * 16 * 9)
        #     print('25 x 25 x 9 would be', 2 * 25 * 25 * 9)  # ignores internal dim, repeats first
        #     print('TF stats gives', flops.total_float_ops)
