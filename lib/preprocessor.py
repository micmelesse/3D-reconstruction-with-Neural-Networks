import tensorflow as tf


def tf_sequence_shuffle(value):
    ret = tf.transpose(value, [1, 0, 2, 3, 4])
    ret = tf.random_shuffle(ret)
    ret = tf.transpose(ret, [1, 0, 2, 3, 4])
    return value


class Preprocessor():
    def __init__(self, X, n_timesteps=None):
        with tf.name_scope("preprocessor"):
            if n_timesteps is None:
                n_timesteps = tf.shape(X)[1]

            # drop alpha channel
            X_dropped_alpha = X[:, :, :, :, 0:3]
            # randomly crop
            X_cropped = tf.map_fn(lambda a: tf.random_crop(
                a, [n_timesteps, 127, 127, 3]), X_dropped_alpha, name="random_crops")
            X_shuffled = tf_sequence_shuffle(X_cropped)
        self.out_tensor = X_shuffled
