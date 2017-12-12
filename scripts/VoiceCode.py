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
        data = DataFromFile(db_dir="../datasets/en-django/")
        files = ["all"]
        data.create_datasets(False, files)
        one_hot = tf.one_hot(self.labels,depth=128,axis=-1)
        logits, sequence = self.model(self.input, self.input_length,self.labels, self.labels_length)
        crossent = tf.nn.softmax_cross_entropy_with_logits(
            labels=one_hot, logits=logits
        )
        train_loss = (tf.reduce_sum(crossent))/batch_size

        optimizer = tf.train.AdadeltaOptimizer(.5)
        train_step = optimizer.minimize(train_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10,keep_checkpoint_every_n_hours=2)

        for e in range(100):
            end_of_epoch = False
            b = 0
            while not end_of_epoch:
                batch, end_of_epoch = data.next_batch(size=batch_size)
                feed_dict = {
                    self.input:batch[0],
                    self.input_length:batch[1],
                    self.labels:batch[2],
                    self.labels_length:batch[3]
                }
                t, loss = sess.run([train_step,train_loss],feed_dict=feed_dict)
                b+=1
                if b%10==0:
                    print(b)
            test_set= data.test_set(batch_size=batch_size)
            for i in range(len(test_set)):
                batch = test_set[i]
                feed_dict = {
                    self.input:batch[0],
                    self.input_length:batch[1],
                    self.labels:batch[2],
                    self.labels_length:batch[3]
                }
                l=sess.run(train_loss,feed_dict=feed_dict)
                loss+=l
            print("Epoch number "+str(e))
            print(loss)
            saver.save(sess, "../models/tmp/base_model",global_step=e)



    def infer():
        pass



def main(args):
    vc = VoiceCode()
    vc.train()

if __name__ == '__main__':
    tf.app.run()
