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
        ff = tf.layers.dense(inputs, self.dim)
        norm = LayerNorm()(inputs,ff)
        return norm


class PositionalEncoding(snt.AbstractModule):
    def __init__(self,name="positional_encoding"):
        super(PositionalEncoding, self).__init__(name=name)


class Encoder(snt.AbstractModule):
    def __init__(self,depth=256,name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.depth = depth

    def _build(self, inputs):
        atten = ScaledDotProductAttention()(inputs,self.depth, self.depth)
        ff = FeedForward()(atten)
        return output

class EncoderStack(snt.AbstractModule):
    def __init__(self,name="encoder_stack"):
        super(EncoderStack, self).__init__(name=name)

class Decoder(snt.AbstractModule):
    def __init__(self,name="decoder"):
        super(Decoder, self).__init__(name=name)

class DecoderStack(snt.AbstractModule):
    def __init__(self,name="decoder_stack"):
        super(DecoderStack, self).__init__(name=name)
