import tensorflow as tf
import sonnet as snt

class C2W(snt.AbstractModule):
    def __init__(self, name="c2w"):
        super(C2W, self).__init__(name=name)

    def _build(self, inputs):
        character_embeddings = tf.get_variable("character_embeddings", [128, 100])
        chars = tf.nn.embedding_lookup(character_embeddings, inputs)
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(300)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(300)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, chars)
        fw_weights = tf.get_variable("forward_weights", [300, 300])
        bw_weights = tf.get_variable("back_weights", [300, 300])
        bias = tf.get_variable("c2w_bias", [300])
        return tf.matmul(state_fw, fw_weights)+tf.matmul(state_bw, bw_weights)+bias

class SelectPredictor(snt.AbstractModule):
    def __init__(self, name="select_predictor"):
        super(SelectPredictor, self).__init__(name=name)

    def _build(self, inputs):
        pass


class PTC(snt.AbstractModule):
    def __init__(self, name="psuedo_to_code"):
        super(PTC, self).__init__(name=name)

    def _build(self, inputs):
        #process inputs with c2w
        #put non numbers into a bi-lstm

        #while the newline
        
