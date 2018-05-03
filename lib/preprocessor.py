import tensorflow as tf
from lib import utils


def shuffle_sequence(value):
    with tf.name_scope("shuffle_sequence"):
        ret = tf.transpose(value, [1, 0, 2, 3, 4])
        ret = tf.random_shuffle(ret)
        ret = tf.transpose(ret, [1, 0, 2, 3, 4])
    return value


class Preprocessor():
    def __init__(self, X):
        with tf.name_scope("preprocessor"):
            params = utils.read_params()
            if params["TRAIN"]["TIME_STEPS"] > 0:
                n_timesteps = params["TRAIN"]["TIME_STEPS"]
            else:
                n_timesteps = tf.shape(X)[1]

            # drop alpha channel
            X_dropped_alpha = X[:, :, :, :, 0:3]
            # randomly crop
            X_cropped = tf.map_fn(lambda a: tf.random_crop(
                a, [n_timesteps, 127, 127, 3]), X_dropped_alpha, name="random_crops")

            if params["TRAIN"]["SHUFFLE_SEQUENCE"]:
                X_shuffled = shuffle_sequence(X_cropped)
                self.out_tensor = X_shuffled
            else:
                self.out_tensor = X_cropped
