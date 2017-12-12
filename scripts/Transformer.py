import tensorflow as tf
import sonnet as snt


class MultiHeadAttention(snt.AbstractModule):
    def __init__(self,name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

class MaskedMultiHeadAttention(snt.AbstractModule):
    def __init__(self,name="masked_multi_head_attention"):
        super(MaskedMultiHeadAttention, self).__init__(name=name)

class FeedForward(snt.AbstractModule):
    def __init__(self,name="feed_forward"):
        super(FeedForward, self).__init__(name=name)

class PositionalEncoding(snt.AbstractModule):
    def __init__(self,name="positional_encoding"):
        super(PositionalEncoding, self).__init__(name=name)

class AddNorm(snt.AbstractModule):
    def __init__(self,name="add_norm"):
        super(AddNorm, self).__init__(name=name)

class Encoder(snt.AbstractModule):
    def __init__(self,name="encoder"):
        super(Encoder, self).__init__(name=name)

class EncoderStack(snt.AbstractModule):
    def __init__(self,name="encoder_stack"):
        super(EncoderStack, self).__init__(name=name)

class Decoder(snt.AbstractModule):
    def __init__(self,name="decoder"):
        super(Decoder, self).__init__(name=name)

class DecoderStack(snt.AbstractModule):
    def __init__(self,name="decoder_stack"):
        super(DecoderStack, self).__init__(name=name)
