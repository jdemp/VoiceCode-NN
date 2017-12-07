import tensorflow as tf
import numpy as np
from ptc import PTC
from data_from_file import DataFromFile

class VoiceCode(object):
    def __init__(self, language="Python"):
        self.language = language

        #placeholders
        self.input = tf.placeholder(shape=[None, None, None], dtype=tf.int32)
        self.input_length = tf.placeholder(shape=[None], dtype=tf.int32)
        self.labels = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.labels_length = tf.placeholder(shape=[None], dtype=tf.int32)

        self.model = PTC()

    def train(self):
        batch_size = 64
        dff = DataFromFile(db_dir="../datasets/en-django/")
        files = ["all"]
        d=dff.create_datasets(False, files)
        dataset = d["all"]

        logits, sequence = self.model(self.input, self.input_length,self.labels, self.labels_length)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits
        )
        train_loss = (tf.reduce_sum(crossent))/batch_size

        optimizer = tf.train.AdamOptimizer(.1)
        train_step = optimizer.minimize(train_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        #dataset = dataset.batch(batch_size)
        #iterator = dataset.make_initializable_iterator()
        #next_batch = iterator.get_next()

        #batch = sess.run(next_batch)
        #print(batch)

    def infer():
        pass



def main(args):
    vc = VoiceCode()
    vc.train()

if __name__ == '__main__':
    tf.app.run()
