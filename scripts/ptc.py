import tensorflow as tf
import sonnet as snt
import numpy as np


class C2W(snt.AbstractModule):
    def __init__(self, size, name="c2w"):
        super(C2W, self).__init__(name=name)
        self.size = size

    def _build(self, inputs):
        batch_size = tf.shape(inputs)[0]
        dtype = tf.float32
        fw_cell = snt.LSTM(self.size, name="lstm_fw")
        bw_cell = snt.LSTM(self.size, name="lstm_bw")
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                    inputs, dtype=tf.float32)
#
        # fw_weights = tf.get_variable("forward_weights", [self.size, self.size])
        # bw_weights = tf.get_variable("back_weights", [self.size, self.size])
         #bias = tf.get_variable("c2w_bias", [self.size])
         #return tf.matmul(state_fw, fw_weights)+tf.matmul(state_bw, bw_weights)+bias
        print(tf.shape(state_fw))
        return state_fw + state_bw


class PTC(snt.AbstractModule):
    def __init__(self, language="Python", use_c2w=True, name="ptc"):
        super(PTC,self).__init__(name=name)
        self.language = language
        self.params = {
            "c2w_size":300,
            "encoder_size":300
        }

    #inputs are [batch size, max_sequence_length, max_word_length]
    def _build(self, inputs):
        embeddings = tf.get_variable("character_embeddings", [128, 100])
        chars = tf.nn.embedding_lookup(embeddings,inputs)
        #chars are [batch size, max_sequence_length, max_word_length, 100]
        c2w = C2W(300)
        # returns [batch size, max max_sequence_length, 300]
        c2w_batch = snt.BatchApply(c2w)
        c2w_words = c2w_batch(chars)


        return c2w_words

        #cell = tf.nn.rnn_cell.LSTMCell(512)
        #attention_mech = tf.contrib.seq2seq.LuongAttention(512, inputs)
        #attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        #    cell, attention_mech, attention_size=256
        #)

def main(args):

    model = PTC()
    p = tf.placeholder(shape=[None,None,None], dtype=tf.int32)
    out = model(p)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    test = np.random.randint(0,10,(2,10,20))
    print(test)
    output = sess.run([out], feed_dict={p:test})
    print(tf.shape(output))


if __name__ == '__main__':
    tf.app.run()
