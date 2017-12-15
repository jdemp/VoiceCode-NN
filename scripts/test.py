import tensorflow as tf
import numpy as np


# inputs = np.random.random_integers(0,100,size=(3,5,128))
# inputs = tf.constant(inputs, dtype=tf.int32)
#
# q = tf.layers.dense(inputs, 128, use_bias=False)
# k = tf.layers.dense(inputs, 128, use_bias=False)
# v = tf.layers.dense(inputs, 128, use_bias=False)
#
# #scale = tf.rsqrt(tf.to_float(tf.shape(q)[2]))
# logits = tf.matmul(q, k, transpose_b=True)
# #shape = tf.shape(logits)
# weights = tf.nn.softmax(tf.cast(logits,tf.float32), name="attention_weights")
# atten = tf.matmul(weights, tf.cast(v,tf.float32))
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
#
# out = sess.run(weights)


ones = tf.ones([5,1])
zeros = tf.zeros([5,4])
final = tf.concat([ones,zeros],1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(final))
