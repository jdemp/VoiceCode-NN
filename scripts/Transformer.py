import tensorflow as tf
import sonnet as snt


class LayerNorm(snt.AbstractModule):
    def __init__(self,name="layer_norm"):
        super(LayerNorm, self).__init__(name=name)

    def _build(self, x, x1):
        add = tf.add(x, x1)
        norm = tf.nn.l2_normalize(add, -1)
        return norm


class ScaledDotProductAttention(snt.AbstractModule):
    def __init__(self,masked=False,name="scaled_dot_product_attention"):
        super(ScaledDotProductAttention, self).__init__(name=name)

    def _build(self, query, key_depth, value_depth, memory=None):
        if memory=None:
            memory=query
        q = tf.layers.dense(query, key_depth, use_bias=False)
        k = tf.layers.dense(memory, key_depth, use_bias=False)
        v = tf.layers.dense(memory, value_depth, use_bias=False)

        scale = tf.rsqrt(tf.to_float(tf.shape(q)[2]))
        logits = tf.matmul(q*scale, k, transpose_b=True)

        weights = tf.nn.softmax(logits, name="attention_weights")
        atten = tf.matmul(weights, v)
        norm = LayerNorm()(query, atten)
        return norm


class FeedForward(snt.AbstractModule):
    def __init__(self,dim=256,name="feed_forward"):
        super(FeedForward, self).__init__(name=name)
        self.dim = dim

    def _build(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        length = shape[1]
        x = tf.reshape(inputs,[batch_size*length,self.dim])
        ff1 = tf.layers.dense(x, self.dim*4, activation=tf.nn.relu)
        ff2 = tf.layers.dense(ff1, self.dim, activation=None)
        y = tf.reshape(ff2, [batch_size,length,self.dim])
        norm = LayerNorm()(inputs,y)
        return norm


class PositionalEncoding(snt.AbstractModule):
    def __init__(self,name="positional_encoding"):
        super(PositionalEncoding, self).__init__(name=name)


    def _build(self, inputs):



class Embedding(snt.AbstractModule):
    def __init__(self,vocab,dim=256,name="embedding"):
        super(Embedding, self).__init__(name=name)
        self.vocab = vocab
        self.dim = dim

    def _build(self,inputs):
        embeddings = tf.get_variable("embeddings",[self.vocab, self.dim])
        lookup = tf.nn.embedding_lookup(embeddings,inputs)
        return lookup

class Encoder(snt.AbstractModule):
    def __init__(self,depth=256,name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.depth = depth

    def _build(self, inputs):
        atten = ScaledDotProductAttention()(inputs,self.depth, self.depth)
        ff = FeedForward()(atten)
        return output

class EncoderStack(snt.AbstractModule):
    def __init__(self,depth=256, layers=2, name="encoder_stack"):
        super(EncoderStack, self).__init__(name=name)
        self.layers = layers
        self.depth = depth

    def _build(self, inputs):
        x = inputs
        for layer in range(self.layers):
            x = Encoder(self.depth)(x)
        return x

class Decoder(snt.AbstractModule):
    def __init__(self,depth=256,name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.depth = depth

    def _build(self, inputs, encoder_output):
        masked_attention = ScaledDotProductAttention()(inputs,self.depth,
                                                            self.depth)
        attention = ScaledDotProductAttention()(masked_attention,self.depth,
                                                    self.depth, encoder_output)
        ff = FeedForward()(attention)
        return ff

class DecoderStack(snt.AbstractModule):
    def __init__(self,depth=256,layers=2, name="decoder_stack"):
        super(DecoderStack, self).__init__(name=name)
        self.layers = layers
        self.depth = depth

    def _build(self, outputs_shifted, encoder_outputs):
        x = outputs_shifted
        for layer in range(self.layers):
            x = Decoder(self.depth)(outputs_shifted, encoder_outputs)
        return x
