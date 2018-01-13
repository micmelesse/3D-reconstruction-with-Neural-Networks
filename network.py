import layers
import tensorflow as tf


class network:
    def __init__(self, learn_rate):
        # place holders
        self.X = tf.placeholder(tf.float32, [None, 24, 137, 137, 4])
        self.Y = tf.placeholder(tf.float32, [None, 32, 32, 32])

        print("encoder network")
        cur_tensor = self.X
        print(cur_tensor.shape)
        self.encoder_outputs = [cur_tensor]
        k_s = [3, 3]
        conv_filter_count = [96, 128, 256, 256, 256, 256]
        for i in range(6):
            k_s = [7, 7] if i is 0 else k_s
            with tf.name_scope("encoding_block"):
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

        print("recurrent module")
        with tf.name_scope("recurrent_module"):
            N, n_x, n_h = 4, 1024, 256
            self.recurrent_module = layers.GRU_R2N2(
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

        print("decoding network")
        self.decoder_outputs = [cur_tensor]
        cur_tensor = layers.unpool3D(cur_tensor)
        print(cur_tensor.shape)
        self.decoder_outputs.append(cur_tensor)

        k_s = [3, 3, 3]
        deconv_filter_count = [128, 128, 128, 64, 32, 2]
        for i in range(2, 4):
            with tf.name_scope("decoding_block"):
                cur_tensor = tf.layers.conv3d(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = layers.unpool3D(cur_tensor)
                cur_tensor = tf.nn.relu(cur_tensor)
                print(cur_tensor.shape)
                self.decoder_outputs.append(cur_tensor)

        for i in range(4, 6):
            with tf.name_scope("decoding_block_without_unpooling"):
                cur_tensor = tf.layers.conv3d(
                    cur_tensor, padding='SAME', filters=deconv_filter_count[i], kernel_size=k_s, activation=None)
                cur_tensor = tf.nn.relu(cur_tensor)
                print(cur_tensor.shape)
                self.decoder_outputs.append(cur_tensor)

        print("predictions")
        softmax_output = tf.nn.softmax(self.decoder_outputs[-1])
        prediction = tf.to_float(tf.argmax(softmax_output, axis=4))
        print(prediction.shape)

        print("losses")
        log_p, log_q = tf.log(softmax_output[:, :, :, :, 0]), tf.log(
            softmax_output[:, :, :, :, 1])
        self.cross_entropies = - tf.reduce_sum(tf.multiply(log_p, self.Y) +
                                               tf.multiply(log_q, 1 - self.Y), [1, 2, 3])
        self.mean_loss = tf.reduce_mean(self.cross_entropies)
        self.optimizing_op = tf.train.GradientDescentOptimizer(
            learning_rate=learn_rate).minimize(tf.reduce_mean(self.mean_loss))
        print(self.cross_entropies.shape)

        print("metrics")
        self.accuracies = tf.reduce_sum(tf.to_float(tf.equal(self.Y, prediction)), axis=[
            1, 2, 3]) / tf.constant(32 * 32 * 32, dtype=tf.float32)  # 32*32*32=32768
        self.mean_accuracy = tf.reduce_mean(self.accuracies)
        print(self.accuracies.shape)
