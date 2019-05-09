import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    def map_fn(x):
        """
        the operation in map_fn should be tensorflow operation
        or it sould be interpretable by tensorflow operation
        """

        return x*2  # "*" can be interpreted as tf.multiply by tensorflow


    dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
    dataset = dataset.map(map_func=lambda x: 2*x)

    iter = dataset.make_one_shot_iterator()

    el = iter.get_next()

    with tf.Session() as sess:
        for i in range(10):
            print(sess.run(el))
