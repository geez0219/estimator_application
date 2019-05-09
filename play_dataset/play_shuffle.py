"""
what shuffle buffer_size means?
if buffer_size = 1, the sequence is doomed to be 0,1,2,3,4,5,6,7,8,9
if buffer_size = n, it will randomly divide the queue it n pieces (the order in each piece is small-to-large)
and the sequence is randomly sample each pieces.

"""


import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
    dataset = dataset.shuffle(2)

    iter = dataset.make_one_shot_iterator()

    el = iter.get_next()

    with tf.Session() as sess:
        for i in range(10):
            print(sess.run(el))
