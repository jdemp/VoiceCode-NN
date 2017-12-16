import tensorflow as tf
import numpy as np
import Transformer
from data_from_file import DataFromFile



class VoiceCode(object):
    def __init__(self):
        self.params ={
            "input_vocab":5000,
            "dim":128,
            "max_out":64,
            "out_vocab":5000,
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
        self.load_vocab()
        # print(self.input_vocab)
        # print(self.output_vocab)


    def model_fn(self, features, labels, mode):
        inputs = features["inputs_num"]
        shape = tf.shape(inputs)
        batch_size = shape[0]
        length = shape[1]
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
        ones = tf.ones([batch_size,1],dtype=tf.int32)
        zeros = tf.zeros([batch_size,1],dtype=tf.int32)
        shifted_labels = tf.concat([ones*3,labels],1)
        decoder_input = output_embedding(shifted_labels)

        decoder_output = decoder(decoder_input,encoder_output)
        logits = tf.layers.dense(decoder_output, self.params["out_vocab"])
        concat_labels = tf.concat([labels,zeros],1)
        softmax_train = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,labels=concat_labels)
        loss = tf.reduce_mean(softmax_train,name="loss")
        train = tf.train.AdamOptimizer(0.01).minimize(loss,
                global_step=tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train)


        softmax = tf.nn.softmax(logits, name="softmax")
        argmax = tf.argmax(softmax,axis=-1,output_type=tf.int32)

        sequence_accuracy = tf.to_float(tf.equal(argmax,concat_labels))
        acc=tf.reduce_mean(sequence_accuracy, -1)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric = {
                "accuracy":tf.metrics.mean(acc)
            }
            return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric)


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
        tf.logging.set_verbosity(tf.logging.INFO)
        dataset = DataFromFile(db_dir="../datasets/en-django/")
        files = ["all"]
        ds = dataset.create_datasets(False, files, self.input_vocab, self.output_vocab)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"inputs_num": ds["inputs"]},
            y=ds["labels"],
            batch_size=32,
            num_epochs=16,
            shuffle=True
        )


        eval_input_fn=tf.estimator.inputs.numpy_input_fn(
            x={"inputs_num": ds["inputs"]},
            y=ds["labels"],
            num_epochs=1,
            shuffle=False
        )

        tensors_to_log = {
            "Loss":"loss"
        }
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=512)

        estimator = tf.estimator.Estimator(
                model_fn=self.model_fn, model_dir="../models/v1/run"
        )

        for i in range(1,33):
            estimator.train(train_input_fn, hooks=[logging_hook])
            print("Finished "+str(i*16)+" epochs")
            print("Evaluating Model")
            ev = estimator.evaluate(eval_input_fn)
            print(ev)


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
            if not word.isdigit():
                self.input_vocab[i_count]=word
                self.input_vocab[word]=i_count
                i_count+=1

        for line in o:
            if i_count == self.params["out_vocab"]:
                break
            word = line.split()[0]
            if not word.isdigit():
                self.output_vocab[i_count]=word
                self.output_vocab[word]=o_count
                o_count+=1


if __name__ == '__main__':
    model = VoiceCode()
    model.train()
