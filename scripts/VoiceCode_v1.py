import tensorflow as tf
import numpy as np
import Transformer


class VoiceCode(object):
    def __init__(self):
        self.params ={
            "input_vocab":5000,
            "dim":128,
            "max_out":32,
            "out_vocab":1000
        }


    def model_fn(self, features, labels, mode):
        inputs = features["inputs_num"]
        inputs_string = features["inputs_string"]


        #generate the position tensor for doing predictor


        #input embedding
        input_embedding = Transformer.Embedding(self.params["input_vocab"],
                                                self.params["dim"])
        encoder_input = input_embedding(inputs)

        #positional encoding

        #enocoder
        encoder = Transformer.EncoderStack(self.params["dim"])
        encoder_output = encoder(encoder_input)

        #decoder
        decoder = Transformer.DecoderStack(self.params["dim"])
        masked_decoder_input = tf.zeros(shape=tf.shape(labels))
        decoder_output = decoder(masked_decoder_input,encoder_output)
        logits = tf.layers.dense(decoder_output, self.params["outvocab"])
        softmax_train = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,labels=labels)
        loss = tf.reduce_mean(softmax_train)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train = tf.train.AdamOptimizer(0.001).minimize(loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train)

        # get decoder working
        #where i is the step of the decoder_loop
        # in [batch size, i]
        # out [batch size, i, dim]
        def process_decode_inputs(i,targets):
            pass

        


        predictions ={
            "probs": tf.nn.softmax(logits,name="sotfmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
