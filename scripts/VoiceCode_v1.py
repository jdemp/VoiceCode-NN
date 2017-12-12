import tensorflow as tf
import numpy as np
import ptc

class VoiceCode(object):
    def __init__(self):
        self.params ={
            "input_vocab":5000,
            "input_embed":128,
            "language_vocab":128,
            "language_embed":64
        }


    def model_fn(self, features, labels, mode):
        inputs_num = features["inputs_num"]
        inputs_string = features["inputs_string"]

        embed = ptc.Embedding(self.params["input_vocab"],
                                self.params["input_embed"])(inputs_num)

        encoder_out, (fw_state,bw_state) = ptc.Encoder(128)(embed)
        decoder = ptc.Decoder()
                            
