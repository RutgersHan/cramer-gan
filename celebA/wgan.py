import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(x * alpha, 0), x)


def dsamp_block(x, nf):
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, nf, [4, 4], [2, 2], padding='same')
    return x


def usamp_block(x, xc, nf):
    _, nh, nw, _ = x.get_shape().as_list()
    if xc is not None:
        x = tf.concat([x, xc], axis=3)
    x = tf.nn.relu(x)
    x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
    x = tf.layers.conv2d(x, nf, [3, 3], [1, 1], padding='same')
    return x


class Discriminator(object):
    def __init__(self):
        self.name = 'celebA/dcgan/d_net'
        self.x_dim = [64, 64, 3]

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            conv1 = tf.layers.conv2d(x, 64, [4, 4], [2, 2])  # 32 * 32 * 64
            conv1 = leaky_relu(conv1)
            conv2 = tf.layers.conv2d(conv1, 128, [4, 4], [2, 2]) # 16 * 16 * 128
            conv2 = leaky_relu(conv2)
            conv3 = tf.layers.conv2d(conv2, 256, [4, 4], [2, 2]) # 8 * 8 * 256
            conv3 = leaky_relu(conv3)
            conv4 = tf.layers.conv2d(conv3, 512, [4, 4], [2, 2]) # 4 * 4 * 512
            conv4 = leaky_relu(conv4)

            xs = conv4.get_shape().as_list()
            x = tf.reshape(conv4, [-1, np.prod(xs[1:])])
            # conv2 = tf.contrib.layers.flatten(conv2) #
            # fc1 = tf.layers.dense(conv2, 1024)
            # fc1 = leaky_relu(fc1)

            fc = tf.layers.dense(x, 1)
            return tf.reshape(fc, [-1])

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.name = 'celebA/dcgan/g_net'
        self.x_dim = [64, 64, 3]

    def __call__(self, x, z, reuse=True):
        x_left = x[:, :, :32, :]
        image64 = tf.concat([x_left, tf.zeros_like(x_left)], axis=2)
        print('image64 shape  ', image64.shape)
        with tf.variable_scope(self.name, reuse=reuse) as vs:
            encoder32 = tf.layers.conv2d(image64, 64, [4, 4], [2, 2], padding='same') # 32 * 32 * 64
            encoder16 = dsamp_block(encoder32, 128)  # 16 * 16 * 128
            encoder8 = dsamp_block(encoder16, 256)  # 8 * 8 * 256
            encoder4 = dsamp_block(encoder8, 512)  # 4 * 4 * 512
            encoder2 = dsamp_block(encoder4, 512)  # 2 * 2 * 512
            encoder1 = dsamp_block(encoder2, 512)  # 1 * 1 * 512

            noise = tf.layers.dense(z, 512)
            noise = tf.reshape(noise, shape=[-1, 1, 1, 512])

            decoder2 = usamp_block(encoder1, noise, 512)  # 2 * 2 * 512
            decoder4 = usamp_block(decoder2, encoder2, 512)  # 4 * 4 * 512
            decoder8 = usamp_block(decoder4, encoder4, 256)  # 8 * 8 * 256
            decoder16 = usamp_block(decoder8, encoder8, 128)  # 16 * 16 * 128
            decoder32 = usamp_block(decoder16, encoder16, 64)  # 32 * 32 * 64
            decoder64 = usamp_block(decoder32, encoder32, 3)  # 64 * 64 * 3
            xg = tf.nn.tanh(decoder64)

            return tf.concat([x_left, xg[:, :, 32:, :]], axis=2)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]