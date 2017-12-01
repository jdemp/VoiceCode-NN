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
        final = tf.layers.dense(concat, 256)

        return final

class Encoder(snt.AbstractModule):
    def __init__(self,units,layers=1,name="encoder"):
        super(Encoder,self).__init__(name=name)
        self.units=units
        self.layers=layers

    def _build(self,inputs):
        fw_cell= tf.nn.rnn_cell.LSTMCell(self.units)
        bw_cell= tf.nn.rnn_cell.LSTMCell(self.units)
        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)
        return tf.concat(encoder_outputs, 2)


class Attention(snt.AbstractModule):
    def __init__(self,units=512,attn_size=256,name="attention"):
        super(Attention,self).__init__(name=name)
        self.units=units
        self.attention_size=attn_size

    def _build(self,inputs):
        cell = tf.nn.rnn_cell.LSTMCell(self.units)
        attention_mech=tf.contrib.seq2seq.LuongAttention(self.units, inputs)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mech,attention_size=self.attention_size
        )

        return attn_cell


# class Decoder(snt.AbstractModule):
#     def __init__(self,name="decoder"):
#         super(Decoder,self).__init(name=name)
#
#     def _build(self,inputs):
#         if mode == "train":
#             helper = tf.contrib.seq2seq.TrainingHelper(
#             input=input_vectors,
#             sequence_length=input_lengths)
#         elif mode == "infer":
#             helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
#             embedding=embedding,
#             start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
#             end_token=END_SYMBOL)



class PTC(snt.AbstractModule):
    def __init__(self, language="Python", use_c2w=True, name="ptc"):
        super(PTC,self).__init__(name=name)
        self.language = language

    #inputs are [batch size, max_sequence_length, max_word_length]
    def _build(self, inputs):
        shape=tf.shape(inputs)
        batch_size = shape[0]
        max_sequence_length = shape[1]
        max_word_length = shape[2]
        #may switch to snt.batchapply at somepoint
        c2w_input = tf.reshape(inputs, [batch_size*max_sequence_length,max_word_length])
        c2w = C2W(256)
        words = c2w(c2w_input)
        sequences = tf.reshape(words, [batch_size, max_sequence_length, 256])

        #encoder
        encoder = Encoder(256)
        encoder_out = encoder(sequences)

        #attention


        return tf.shape(encoder_out)

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
