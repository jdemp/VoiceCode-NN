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


zeros = tf.ones([3,5,5])*-1e9
mask = tf.matrix_band_part(zeros,-1,0)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(mask))
