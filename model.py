from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange
from scipy.misc import imresize
from subpixel import PS

from ops import *
from utils import *

def doresize(x, shape):
    x = np.copy((x+1.)*127.5).astype("uint8")
    y = imresize(x, shape)
    return y

class DCGAN(object):
    def __init__(self, sess, image_size=128, is_crop=True,
                 batch_size=64, image_shape=[128, 128, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_size = 32
        self.sample_size = batch_size
        self.image_shape = image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3],
                                    name='real_images')
        try:
            self.up_inputs = tf.image.resize_images(self.inputs, self.image_shape[0], self.image_shape[1], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        except ValueError:
            # newer versions of tensorflow
            self.up_inputs = tf.image.resize_images(self.inputs, [self.image_shape[0], self.image_shape[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape,
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,
                                        name='sample_images')

        self.G = self.generator(self.inputs)

        self.G_sum = tf.summary.image("G", self.G)

        self.g_loss = tf.reduce_mean(tf.square(self.images-self.G))

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def infer(self, config):
        """Inference"""

        # first setup validation data
        data = sorted(glob(os.path.join("./data", config.infer_dataset, "valid", "*.jpg")))

        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        self.g_sum = tf.summary.merge([self.G_sum, self.g_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        if config.experimental:
            dataset_id = "expr_" + config.infer_dataset

            sample_files = data[0:self.sample_size]
            sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, product_size=self.image_size * 2) for sample_file in sample_files]
            sample_inputs = [doresize(xx, [self.input_size*2,]*2) for xx in sample]
            sample_images = np.array(sample).astype(np.float32)
            sample_input_images = np.array(sample_inputs).astype(np.float32)

            sample_input_images_bucubic = np.array([imresize(img, [self.image_size*2, self.image_size*2]) for img in sample_input_images])
            print(sample_input_images_bucubic.shape)
            save_images(sample_input_images_bucubic, [8, 8], './samples/' + dataset_id + '_small_bicubic.png')

            save_images(sample_input_images, [8, 8], './samples/inputs_' + dataset_id + '_small.png')
            save_images(sample_images, [8, 8], './samples/' + dataset_id + '_reference.png')

            x00 = np.array([i[::2, ::2] for i in sample_inputs])
            x01 = np.array([i[::2, 1::2] for i in sample_inputs])
            x10 = np.array([i[1::2, ::2] for i in sample_inputs])
            x11 = np.array([i[1::2, 1::2] for i in sample_inputs])

            sample_images_x00 = np.array([i[::2, ::2] for i in sample_images])
            sample_images_x01 = np.array([i[::2, 1::2] for i in sample_images])
            sample_images_x10 = np.array([i[1::2, ::2] for i in sample_images])
            sample_images_x11 = np.array([i[1::2, 1::2] for i in sample_images])

            print(np.array(sample).shape)
            print(x00.shape)

            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            print "Eww"

            samples_x00, g_loss_x00, up_inputs_x00 = self.sess.run(
                [self.G, self.g_loss, self.up_inputs],
                feed_dict={self.inputs: x00, self.images: sample_images_x00}
            )
            samples_x01, g_loss_x01, up_inputs_x01 = self.sess.run(
                [self.G, self.g_loss, self.up_inputs],
                feed_dict={self.inputs: x01, self.images: sample_images_x01}
            )
            samples_x10, g_loss_x10, up_inputs_x10 = self.sess.run(
                [self.G, self.g_loss, self.up_inputs],
                feed_dict={self.inputs: x10, self.images: sample_images_x10}
            )
            samples_x11, g_loss_x11, up_inputs_x11 = self.sess.run(
                [self.G, self.g_loss, self.up_inputs],
                feed_dict={self.inputs: x11, self.images: sample_images_x11}
            )

            avg_loss = (g_loss_x00 + g_loss_x01 + g_loss_x10 + g_loss_x11) / 4.0

            up_inputs = np.zeros((self.sample_size, self.image_size * 2, self.image_size * 2, 3))
            samples = np.zeros((self.sample_size, self.image_size * 2, self.image_size * 2, 3))

            for sample in xrange(0, self.sample_size):
                for w_idx in xrange(0, self.image_size):
                    for h_idx in xrange(0, self.image_size):
                        up_inputs[sample, w_idx * 2, h_idx * 2] = up_inputs_x00[sample, w_idx, h_idx]
                        up_inputs[sample, w_idx * 2, h_idx * 2 + 1] = up_inputs_x01[sample, w_idx, h_idx]
                        up_inputs[sample, w_idx * 2 + 1, h_idx * 2] = up_inputs_x10[sample, w_idx, h_idx]
                        up_inputs[sample, w_idx * 2 + 1, h_idx * 2 + 1] = up_inputs_x11[sample, w_idx, h_idx]
                        samples[sample, w_idx * 2, h_idx * 2] = samples_x00[sample, w_idx, h_idx]
                        samples[sample, w_idx * 2, h_idx * 2 + 1] = samples_x01[sample, w_idx, h_idx]
                        samples[sample, w_idx * 2 + 1, h_idx * 2] = samples_x10[sample, w_idx, h_idx]
                        samples[sample, w_idx * 2 + 1, h_idx * 2 + 1] = samples_x11[sample, w_idx, h_idx]

            print(up_inputs.shape)

            save_images(up_inputs, [8, 8], './samples/' + dataset_id + '_inputs.png')
            save_images(samples, [8, 8],
                        './samples/' + dataset_id + '_valid.png')
            print("[Sample] g_loss: %.8f %.8f %.8f %.8f avg %.8f" % (g_loss_x00, g_loss_x01, g_loss_x10, g_loss_x11, avg_loss))
            return
        else:
            print "Eww2"

        dataset_id = "infer_" + config.infer_dataset

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_inputs = [doresize(xx, [self.input_size,]*2) for xx in sample]
        sample_images = np.array(sample).astype(np.float32)
        sample_input_images = np.array(sample_inputs).astype(np.float32)

        save_images(sample_input_images, [8, 8], './samples/inputs_' + dataset_id + '_small.png')
        save_images(sample_images, [8, 8], './samples/' + dataset_id + '_reference.png')

        sample_input_images_bucubic = np.array([imresize(img, [self.image_size, self.image_size]) for img in sample_input_images])
        print(sample_input_images_bucubic.shape)
        save_images(sample_input_images_bucubic, [8, 8], './samples/' + dataset_id + '_small_bicubic.png')
            
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        samples, g_loss, up_inputs = self.sess.run(
            [self.G, self.g_loss, self.up_inputs],
            feed_dict={self.inputs: sample_input_images, self.images: sample_images}
        )
        save_images(up_inputs, [8, 8], './samples/' + dataset_id + '_inputs.png')
        save_images(samples, [8, 8],
                    './samples/' + dataset_id + '_valid.png')
        print("[Sample] g_loss: %.8f" % (g_loss))

    def train(self, config):
        """Train DCGAN"""
        # first setup validation data
        data = sorted(glob(os.path.join("./data", config.dataset, "valid", "*.jpg")))

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        self.g_sum = tf.summary.merge([self.G_sum, self.g_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_inputs = [doresize(xx, [self.input_size,]*2) for xx in sample]
        sample_images = np.array(sample).astype(np.float32)
        sample_input_images = np.array(sample_inputs).astype(np.float32)

        save_images(sample_input_images, [8, 8], './samples/inputs_small.png')
        save_images(sample_images, [8, 8], './samples/reference.png')

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # we only save the validation inputs once
        have_saved_inputs = False

        for epoch in xrange(config.epoch):
            data = sorted(glob(os.path.join("./data", config.dataset, "train", "*.jpg")))
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
                input_batch = [doresize(xx, [self.input_size,]*2) for xx in batch]
                batch_images = np.array(batch).astype(np.float32)
                batch_inputs = np.array(input_batch).astype(np.float32)

                # Update G network
                _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                    feed_dict={ self.inputs: batch_inputs, self.images: batch_images })
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errG))

                if np.mod(counter, 100) == 1:
                    samples, g_loss, up_inputs = self.sess.run(
                        [self.G, self.g_loss, self.up_inputs],
                        feed_dict={self.inputs: sample_input_images, self.images: sample_images}
                    )
                    if not have_saved_inputs:
                        save_images(up_inputs, [8, 8], './samples/inputs.png')
                        have_saved_inputs = True
                    save_images(samples, [8, 8],
                                './samples/valid_%s_%s.png' % (epoch, idx))
                    print("[Sample] g_loss: %.8f" % (g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def generator(self, z):
        # project `z` and reshape
        self.h0, self.h0_w, self.h0_b = deconv2d(z, [self.batch_size, 32, 32, self.gf_dim], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0', with_w=True)
        h0 = lrelu(self.h0)

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, 32, 32, self.gf_dim], name='g_h1', d_h=1, d_w=1, with_w=True)
        h1 = lrelu(self.h1)

        h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, 32, 32, 3*16], d_h=1, d_w=1, name='g_h2', with_w=True)
        h2 = PS(h2, 4, color=True)

        return tf.nn.tanh(h2)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def convertToMPS(self, output_dir):
        output_dir = os.path.join(output_dir, '')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # TF ORDERING
        # (K_Y, K_X, IN, OUT)
        #
        # MPS ORDERING
        #
        # It is `(IN, K_Y, K_X, OUT)` for MPSCNNConvolutionTranspose, instead of
        # `(OUT, K_Y, K_X, IN)` which is for MPSCNNConvolution.

        with open(output_dir + 'b_conv1', 'w') as f:
            f.write(self.sess.run(self.h0_b).tobytes())
        with open(output_dir + 'b_conv2', 'w') as f:
            f.write(self.sess.run(self.h1_b).tobytes())
        with open(output_dir + 'b_conv3', 'w') as f:
            f.write(self.sess.run(self.h2_b).tobytes())
        with open(output_dir + 'w_conv1', 'w') as f:
            w_h0_mps = tf.transpose(self.h0_w, [2, 0, 1, 3])
            w_h0_mps = tf.reverse(w_h0_mps, axis=(1, 2))
            f.write(self.sess.run(w_h0_mps).tobytes())
        with open(output_dir + 'w_conv2', 'w') as f:
            w_h1_mps = tf.transpose(self.h1_w, [2, 0, 1, 3])
            w_h1_mps = tf.reverse(w_h1_mps, axis=(1, 2))
            f.write(self.sess.run(w_h1_mps).tobytes())
        with open(output_dir + 'w_conv3', 'w') as f:
            w_h2_mps = tf.transpose(self.h2_w, [2, 0, 1, 3])
            w_h2_mps = tf.reverse(w_h2_mps, axis=(1, 2))
            f.write(self.sess.run(w_h2_mps).tobytes())
