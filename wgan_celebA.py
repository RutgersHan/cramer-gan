import os
import time
import argparse
import importlib
import tensorflow as tf
from scipy.misc import imsave
from visualize import *


def _safer_norm(tensor, axis=None, keep_dims=False, epsilon=1e-5):
  sq = tf.square(tensor)
  squares = tf.reduce_sum(sq, axis=axis, keep_dims=keep_dims)
  return tf.sqrt(squares + epsilon)


class WGAN(object):
    def __init__(self, g_net, d_net, train_sampler, val_sampler, z_sampler, data, model, save_dir, scale=10.0):
        self.save_dir = save_dir
        self.model = model
        self.data = data
        self.d_net = d_net
        self.g_net = g_net
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.z_sampler = z_sampler
        self.x_dim = d_net.x_dim
        self.z_dim = self.g_net.z_dim

        self.x = tf.placeholder(tf.float32, [None] + self.x_dim, name='x')
        self.z1 = tf.placeholder(tf.float32, [None, self.z_dim], name='z1')
        self.x1_ = self.g_net(self.x, self.z1, reuse=False)

        h_real = d_net(self.x, reuse=False)
        h_generated1 = d_net(self.x1_)

        self.g_loss = -tf.reduce_mean(h_generated1)
        self.d_loss = tf.reduce_mean(h_generated1) - tf.reduce_mean(h_real)

        # interpolate real and generated samples
        epsilon = tf.random_uniform([], 0.0, 1.0)
        # Using x and x1_ for the x_hat
        # and using the corresponding h_real and h_generated1
        # in the _gp_critic.
        x_hat = epsilon * self.x + (1 - epsilon) * self.x1_
        h_interpolates = d_net(x_hat)

        ddx = tf.gradients(h_interpolates, x_hat)[0]
        print(ddx.get_shape().as_list())
        # ddx = tf.norm(ddx, axis=1)
        ddx = _safer_norm(ddx, axis=1)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        self.ddx = ddx

        self.d_loss_all = self.d_loss + self.ddx

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss_all, var_list=self.d_net.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=128, num_batches=1000000):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            #if t % 500 == 0 or t < 25:
            #     d_iters = 100

            for _ in range(0, d_iters):
                bx = self.train_sampler(batch_size)
                bz1 = self.z_sampler(batch_size, self.z_dim)
                _, d_loss, ddx = self.sess.run([self.d_adam, self.d_loss, self.ddx],
                                               feed_dict={self.x: bx, self.z1: bz1})
                # print('d_loss', d_loss, 'ddx', ddx)
            bx = self.train_sampler(batch_size)
            bz1 = self.z_sampler(batch_size, self.z_dim)
            _, g_loss = self.sess.run([self.g_adam, self.g_loss], feed_dict={self.z1: bz1, self.x: bx})
            # print('g_loss', g_loss)

            if t % 100 == 0:
                bx = self.train_sampler(batch_size)
                bz1 = self.z_sampler(batch_size, self.z_dim)


                d_loss, ddx_loss = self.sess.run(
                    [self.d_loss, self.ddx], feed_dict={self.x: bx, self.z1: bz1}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z1: bz1, self.x: bx}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] ddx_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss, ddx_loss, g_loss))

            if t % 100 == 0:
                bz1 = self.z_sampler(batch_size, self.z_dim)
                bx_train = self.train_sampler(batch_size)
                bx_val = self.val_sampler(batch_size)

                train_img = self.sess.run(self.x1_, feed_dict={self.z1: bz1,
                                                               self.x: bx_train})
                val_img = self.sess.run(self.x1_, feed_dict={self.z1: bz1,
                                                             self.x: bx_val})
                val_img = val_img[0:48]

                img_tile_val = img_tile(val_img, tile_shape=[6, 8], aspect_ratio=1.0, border_color=1.0,
                                             stretch=False)
                save_tile_img(img_tile_val, os.path.join(self.save_dir, 'sample_val%d.png' % (t/100)))
                img_tile_train = img_tile(train_img,  aspect_ratio=1.0, border_color=1.0,
                                             stretch=False)
                save_tile_img(img_tile_train, os.path.join(self.save_dir, 'sample%d.png' % (t/100)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='celebA')
    parser.add_argument('--model', type=str, default='wgan')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='/home/hanzhang/Result/celebA_wgan')
    args = parser.parse_args()
    args.save_dir = args.save_dir + '_' + args.gpus

    try:
        os.stat(args.save_dir)
    except:
        os.makedirs(args.save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    train_sampler = data.TrainDataSampler()
    val_sampler = data.ValDataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    cgan = WGAN(g_net, d_net, train_sampler, val_sampler, zs, args.data, args.model, args.save_dir)
    cgan.train()
