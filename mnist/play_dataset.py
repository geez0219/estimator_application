import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # x_train = {"pixels": x_train/255.0}
    x_test = {"pixels": x_test / 255.0}
    x_train = {"pixels": x_train / 255.0}

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    def train_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size=batch_size)

        return dataset


    dataset = train_input_fn(x_train, y_train, 128)
    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()

    with tf.Session() as sess:
        a = sess.run(el)
        print()