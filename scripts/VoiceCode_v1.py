import tensorflow as tf
import numpy as np
import Transformer
from data_from_file import DataFromFile


class VoiceCode(object):
    def __init__(self):
        self.params ={
            "input_vocab":2500,
            "dim":128,
            "max_out":32,
            "out_vocab":2500,
            "input vocab file": "../vocab/inputs_python.voc",
            "output vocab file": "../vocab/code.voc"
        }
        self.input_vocab = {
            "PAD":0,
            0:"PAD",
            "UNK":1,
            1:"UNK",
            "NUM":2,
            2:"NUM"

        }
        self.output_vocab = {
            "PAD":0,
            0:"PAD",
            "UNK":1,
            1:"UNK",
            "NUM":2,
            2:"NUM",
            "<start>":3,
            3:"<start>",
            "<end>":4,
            4:"<end>"
        }


    def model_fn(self, features, labels, mode):
        inputs = features["inputs_num"]
        #inputs_string = features["inputs_string"]


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
        output_embedding = Transformer.Embedding(self.params["out_vocab"],
                                                self.params["dim"])
        decoder = Transformer.DecoderStack(self.params["dim"])
        #masked_decoder_input = tf.zeros(shape=tf.shape(labels))

        if mode == tf.estimator.ModeKeys.TRAIN:
            decoder_input = output_embedding(labels)
            decoder_output = decoder(decoder_input,encoder_output)
            logits = tf.layers.dense(decoder_output, self.params["outvocab"])
            softmax_train = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,labels=labels)
            loss = tf.reduce_mean(softmax_train)
            train = tf.train.AdamOptimizer(0.001).minimize(loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train)

        if mode == "test":
            decoder_input = output_embedding(labels)
            decoder_output = decoder(decoder_input,encoder_output)
            logits = tf.layers.dense(decoder_output, self.params["out_vocab"])
            return tf.shape(logits)

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

    def train(self):
        dataset = DataFromFile()
        dataset = DataFromFile(db_dir="../datasets/en-django/")
        files = ["all"]
        dataset.create_datasets(False, files, self.input_vocab, self.output_vocab)


    def test(self):
        inputs = np.random.random_integers(0,100,size=(3,7))
        targets = np.random.random_integers(0,10,size=(3,10))
        features = {
            "inputs_num":tf.constant(inputs,dtype=tf.int32)
        }
        labels = tf.constant(targets, dtype=tf.int32)
        logits = self.model_fn(features,labels,"test")

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        res = sess.run(logits)
        print(res)

    def load_vocab(self):
        i = open(self.params["input vocab file"], mode='r')
        o = open(self.params["output vocab file"], mode='r')

        i_count = 1
        o_count = 3
        for line in i:
            if i_count == self.params["input_vocab"]:
                break
            word = line.split()[0]
            if not w.isdigit():
                self.input_vocab[i_count]=word
                self.input_vocab[word]=i_count
                i_count+=1

        for line in i:
            if i_count == self.params["input_vocab"]:
                break
            word = line.split()[0]
            if not w.isdigit():
                self.output_vocab[i_count]=word
                self.output_vocab[word]=o_count
                o_count+=1


if __name__ == '__main__':
    model = VoiceCode()
    model.train()
