import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    feature = np.array([[i, i] for i in range(10)])
    label = np.array([[i] for i in range(10)])
    dataset = tf.data.Dataset.from_tensor_slices((feature, label)).shuffle(10).repeat().batch(32)

    iter = dataset.make_one_shot_iterator()
    iter2 = dataset.make_one_shot_iterator()
    el = iter.get_next()
    el2 = iter2.get_next()

    with tf.Session() as sess:
        a = el[0] + el2[0]
        tf.summary.FileWriter("tensorboard", sess.graph)





