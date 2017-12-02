import tensorflow as tf
import sonnet as snt
import numpy as np


class C2W(snt.AbstractModule):
    def __init__(self, size, name="c2w"):
        super(C2W, self).__init__(name=name)
        self.size = size

    #takes [words, max_word_length,embedding size]
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
        return tf.concat(encoder_outputs, 2), encoder_state


class Attention(snt.AbstractModule):
    def __init__(self,name="attention"):
        super(Attention,self).__init__(name=name)

    # inputs:[batch,length,512] h:[batch,512]
    def _build(self,inputs, h):
         pass


class Decoder(snt.AbstractModule):
    def __init__(self,units=512,attn_size=256,mode="train",name="decoder"):
        super(Decoder,self).__init__(name=name)
        self.units=units
        self.attention_size=attn_size
        self.mode=mode

    def _build(self,inputs,input_lengths,labels=None,labels_lengths=None):
        batch_size = tf.shape(inputs)[0]

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.units)
        attention_mech=tf.contrib.seq2seq.BahdanauAttention(self.units, inputs, memory_sequence_length=input_lengths)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mech, attention_layer_size=self.units/2
        )

        embeddings = tf.get_variable("embeddings",[128,100])
        decoder_emb_inp = tf.nn.embedding_lookup(embeddings, labels)
        if self.mode == "train":
            helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_emb_inp,
            sequence_length=labels_lengths)
        # elif mode == "infer":
        #     helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #     embedding=embedding,
        #     start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
        #     end_token=END_SYMBOL)
        initial_state=attn_cell.zero_state(batch_size, tf.float32)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attn_cell,
            helper = helper,
            initial_state=initial_state
        )
        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=20
        )
        return final_outputs, final_state, final_sequence_lengths


class PTC(snt.AbstractModule):
    def __init__(self, language="Python", use_c2w=True, name="ptc"):
        super(PTC,self).__init__(name=name)
        self.language = language

    #inputs are [batch size, max_sequence_length, max_word_length]
    def _build(self, inputs,input_lengths,labels,labels_lengths):
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
        encoder_out, (fw_state,bw_state) = encoder(sequences)

        #attention
        attention = Decoder()
        decoder_output, decoder_state, final_sequences = attention(encoder_out,input_lengths,labels, labels_lengths)

        return tf.shape(decoder_output.rnn_output)


def main(args):

    model = PTC()
    p = tf.placeholder(shape=[None,None,None], dtype=tf.int32)
    labels = tf.placeholder(shape=[None,None],dtype=tf.int32)
    labels_len = tf.placeholder(shape=[None],dtype=tf.int32)
    p_l = tf.placeholder(shape=[None],dtype=tf.int32)
    input_lengths = np.array([10,10,10])
    out = model(p,p_l,labels,labels_len)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    test = np.random.randint(0,10,(3,10,25))
    test_lengths = np.array([15,15,15])
    test_labels = np.random.randint(0,50,(3,15))
    #print(test)
    output = sess.run([out], feed_dict={p:test,labels:test_labels,p_l:input_lengths,labels_len:test_lengths})
    print(output)
    #print(tf.shape(output))


if __name__ == '__main__':
    tf.app.run()
