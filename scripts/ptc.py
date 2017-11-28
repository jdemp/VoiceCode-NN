import tensorflow as tf

class PTC(object):
    def __init__(self, language="Python", use_c2w=True):
        self.language = language
        self.use_c2w = use_c2w
        self.input = tf.placeholder(tf.string, [None, None], name="input")
        self.model = self._build_model()

    def _build_c2w(self, inputs):
        byte_input = tf.decode_raw(inputs, tf.unit8, name="convert_input")
        character_embeddings = tf.get_variable("character_embeddings", [128, 100])
        chars = tf.nn.embedding_lookup(character_embeddings, inputs)
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(300)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(300)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, chars)
        fw_weights = tf.get_variable("forward_weights", [300, 300])
        bw_weights = tf.get_variable("back_weights", [300, 300])
        bias = tf.get_variable("c2w_bias", [300])
        return tf.matmul(state_fw, fw_weights)+tf.matmul(state_bw, bw_weights)+bias

    def _build_embeddings(self, inputs):
        if self.use_c2w:
            return self._build_c2w

    def _build_encoder(self, inputs):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(300)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(300)
        (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, chars, dtype=tf.float32)
        output = tf.concat([fw_out, bw_out], axis=2)
        print(output.get_shape())
        return output

    def _build_decoder(self, inputs):
        #
        pass

    def _build_model(self):
        #figure out how to handle variable passthrough

        #embed each word & put through bi-lstm
        embedding = self._build_embeddings
        encoder = self._build_encoder
        return encoder

    def get_model(self):
        return self.model

def main(args):

    model = PTC()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

if __name__ == '__main__':
    tf.app.run()
        # tanh() will have to get the state from the decoder
        # linear()
        # softmax
        # z = linear*softmax (should have size of 300)
