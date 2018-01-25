import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lib.recurrent_module as recurrent_module
import lib.utils as utils

# Recurrent Reconstruction Neural Network (R2N2)
class R2N2:
    def __init__(self, learn_rate):
        # place holders
        self.X = tf.placeholder(tf.float32, [None, 24, 137, 137, 4])
        self.Y = tf.placeholder(tf.uint8, [None, 32, 32, 32])

        print("encoder_network")
        with tf.name_scope("encoder_network"):
            cur_tensor = self.X
            print(cur_tensor.shape)
            self.encoder_outputs = [cur_tensor]
            k_s = [3, 3]
            conv_filter_count = [96, 128, 256, 256, 256, 256]
            for i in range(6):
                k_s = [7, 7] if i is 0 else k_s
                cur_tensor = tf.map_fn(lambda a: tf.layers.conv2d(
                    a, filters=conv_filter_count[i], padding='SAME', kernel_size=k_s, activation=None),  cur_tensor)
                cur_tensor = tf.map_fn(
                    lambda a: tf.layers.max_pooling2d(a, 2, 2),  cur_tensor)
                cur_tensor = tf.map_fn(tf.nn.relu,  cur_tensor)
                print(cur_tensor.shape)
                self.encoder_outputs.append(cur_tensor)

            cur_tensor = tf.map_fn(tf.contrib.layers.flatten,  cur_tensor)
            cur_tensor = tf.map_fn(lambda a: tf.contrib.layers.fully_connected(
                a, 1024, activation_fn=None), cur_tensor)
            self.encoder_outputs.append(cur_tensor)
            print(cur_tensor.shape)

        print("recurrent_module")
        with tf.name_scope("recurrent_module"):
            N, n_x, n_h = 4, 1024, 256
            self.recurrent_module = recurrent_module.GRU_GRID_2(
                n_cells=N, n_input=n_x, n_hidden_state=n_h)

            self.hidden_state_list = []  # initial hidden state
            hidden_state = tf.zeros([1, 4, 4, 4, 256])

            for t in range(24):  # feed batches of seqeuences
                fc_batch_t = cur_tensor[:, t, :]
                self.hidden_state_list.append(hidden_state)
                hidden_state = self.recurrent_module.call(
                    fc_batch_t, hidden_state)
            print(hidden_state.shape)
        cur_tensor = hidden_state

        print("decoder_network")
        with tf.name_scope("decoder_network"):
            self.decoder_outputs = [cur_tensor]
            cur_tensor = utils.unpool3D(cur_tensor)
            print(cur_tensor.shape)
            self.decoder_outputs.append(cur_tensor)

            k_s = [3, 3, 3]
            deconv_filter_count = [128, 128, 128, 64, 32, 2]
            for i in range(2, 4):
                cur_tensor = tf.layers.conv3d(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = utils.unpool3D(cur_tensor)
                cur_tensor = tf.nn.relu(cur_tensor)
                print(cur_tensor.shape)
                self.decoder_outputs.append(cur_tensor)

            for i in range(4, 6):
                cur_tensor = tf.layers.conv3d(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = tf.nn.relu(cur_tensor)
                print(cur_tensor.shape)
                self.decoder_outputs.append(cur_tensor)

        print("softmax_output")
        self.softmax_output = tf.map_fn(lambda a: tf.nn.softmax(a), cur_tensor)
        print(self.softmax_output.shape)
        print("prediction")
        self.prediction = tf.argmax(self.softmax_output, axis=4)
        print(self.prediction.shape)

        print("losses")
        self.cross_entropies = -tf.reduce_sum(tf.multiply(
            tf.log(self.softmax_output), tf.one_hot(self.Y, 2)), axis=[1, 2, 3, 4])
        self.mean_loss = tf.reduce_mean(self.cross_entropies)
        self.optimizing_op = tf.train.GradientDescentOptimizer(
            learning_rate=learn_rate).minimize(self.mean_loss)
        print(self.cross_entropies.shape)

        print("metrics")
        self.accuracies = tf.reduce_sum(tf.to_float(tf.equal(tf.to_int64(self.Y), self.prediction)), axis=[
            1, 2, 3]) / tf.constant(32 * 32 * 32, dtype=tf.float32)  # 32*32*32=32768
        self.mean_accuracy = tf.reduce_mean(self.accuracies)
        print(self.accuracies.shape)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()

    def plot(self, plot_dir, loss, accuracy):
        plt.plot((np.array(loss)).flatten())
        plt.savefig("{}/loss.png".format(plot_dir),
                    bbox_inches='tight')
        plt.clf()
        plt.plot((np.array(accuracy)).flatten())
        plt.ylim(0, 1)
        plt.savefig("{}/accuracy.png".format(plot_dir),
                    bbox_inches='tight')
        plt.clf()

    def save(self, save_dir, loss, accuracy):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        np.save("{}/losses".format(save_dir), np.array(loss))
        np.save("{}/accs".format(save_dir), np.array(accuracy))
        self.saver.save(self.sess, "{}/model.ckpt".format(save_dir))
        self.plot(save_dir, loss, accuracy)

    def vis(self, log_dir="./log"):  # tensorboard/ vis tools
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(self.sess.graph)

    def filters(self, fd):
        return self.sess.run(self.sess, fd)

    def state(self, fd):
        return self.sess.run([self.encoder_outputs, self.hidden_state_list, self.decoder_outputs], fd)
