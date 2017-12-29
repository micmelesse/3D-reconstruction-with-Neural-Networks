import os
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    trainer = Trainer()
    print(trainer.recent_batch())


class Trainer:
    def __init__(self):
        self.root_train_dir = "./train_dir"
        self.cur_time = str(datetime.now()).translate({ord(" "): "_"})

        if not os.path.isdir(self.root_train_dir):
            os.makedirs(self.root_train_dir)

    def recent_training_session(self):
        train_sess_dir = sorted(os.listdir(self.root_train_dir))[-1]
        return os.path.join(self.root_train_dir, train_sess_dir)

    def recent_epoch(self):
        train_sess_dir = self.recent_training_session()
        epoch_dir = sorted(os.listdir(train_sess_dir))[-1]
        return os.path.join(train_sess_dir, epoch_dir)

    def recent_batch(self):
        epoch_dir = self.recent_epoch()
        batch_dir = sorted(os.listdir(epoch_dir))[-1]
        return os.path.join(epoch_dir, batch_dir)


def run():
    pass


""" 3d voxel-wise softmax """


def evaluate(prediction, y):
    y_hat = tf.nn.softmax(prediction)
    p = y_hat[:, :, :, :, 0]
    q = y_hat[:, :, :, :, 1]
    loss_voxel = tf.reduce_mean(tf.reduce_sum(tf.multiply(
        tf.log(p), y) + tf.multiply(tf.log(q), 1 - y), [1, 2, 3]))
    return tf.train.GradientDescentOptimizer(0.5).minimize(loss_voxel)


""" record training session step """


def record(sess, loss, rec_dir):

    fig = plt.figure()
    plt.plot(loss)
    plt.savefig("{}/loss.png".format(rec_dir), bbox_inches='tight')
    save_path = saver.save(sess, "{}/model.ckpt".format(rec_dir))
    plt.close()


if __name__ == '__main__':
    main()
