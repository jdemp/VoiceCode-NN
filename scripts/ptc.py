import tensorflow as tf
import sonnet as snt
import numpy as np


class C2W(snt.AbstractModule):
    def __init__(self, size, name="c2w"):
        super(C2W, self).__init__(name=name)
        self.size = size

    #takes [words, max_word_length]
    def _build(self, inputs):
        embed = tf.contrib.layers.embed_sequence(inputs,128,100)
        fw_cell = snt.LSTM(self.size, name="lstm_fw")
        bw_cell = snt.LSTM(self.size, name="lstm_bw")
        (a,b), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                    embed, dtype=tf.float32)
        concat = tf.concat([a[:,-1,:], b[:,-1,:]], 1)
        final = tf.layers.dense(concat, 300)

        return final

class Encoder(snt.AbstractModule):
    def __init__(self,units,layers=1,name="encoder"):
        super(Encoder,self).__init(name=name)
        self.units=units
        self.layers=layers

    def _build(self,inputs):
        fw_cell= tf.rnn.rnn_cell.LSTMCell(self.units)
        bw_cell= tf.rnn.rnn_cell.LSTMCell(self.units)
        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs)
        return tf.concat(encoder_outputs, 2)


class Attention(snt.Module):
    def __init__(self,units=256,name="attention"):
        super(Attention,self).__init(name=name)
        self.units=units

    def _build(self,inputs):




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
        shape=tf.shape(inputs)
        batch_size = shape[0]
        max_sequence_length = shape[1]
        max_word_length = shape[2]
        #may switch to snt.batchapply at somepoint
        c2w_input = tf.reshape(inputs, [batch_size*max_sequence_length,max_word_length])
        c2w = C2W(300)
        words = c2w(c2w_input)
        sequences = tf.reshape(words, [batch_size, max_sequence_length, 300])


        


        return sequences

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
    test = np.random.randint(0,10,(2,10,25))
    #print(test)
    output = sess.run([out], feed_dict={p:test})
    print(output)
    #print(tf.shape(output))


if __name__ == '__main__':
    tf.app.run()
