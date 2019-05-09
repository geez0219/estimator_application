import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    features = np.array([[i, i] for i in range(10)])
    labels = np.array([[i] for i in range(10)])

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    iter = dataset.make_one_shot_iterator()
    iter2 = dataset.make_initializable_iterator()
    # iter2 = dataset.make_one_shot_iterator()

    el = iter.get_next()
    el2 = iter2.get_next()

    with tf.Session() as sess:
        print(sess.run(el))
        print(sess.run(el))
        print(sess.run(el))
        print(sess.run(el))
        sess.run(iter.make_initializer(dataset=dataset))
        print(sess.run(el))
        tf.summary.FileWriter("tensorboard", sess.graph)
