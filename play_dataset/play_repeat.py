"""
for instance dataset.repeat(2), the sequence is 0,1,2,0,1,2

"""


import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    dataset = tf.data.Dataset.from_tensor_slices(np.arange(3))
    dataset = dataset.repeat(2)

    iter = dataset.make_one_shot_iterator()

    el = iter.get_next()

    with tf.Session() as sess:
        for i in range(20):
            print(sess.run(el))
