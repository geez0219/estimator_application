import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def serialize_example(features, labels):
        assert features.shape == (28*28,)
        feature = {
            "pixel": tf.train.Feature(float_list=tf.train.FloatList(value=features)),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example.SerializeToString()

    with tf.python_io.TFRecordWriter("mnist_train.tfrecord") as writer:
        for i in range(x_train.shape[0]):
            writer.write(serialize_example(x_train[i].reshape([28*28]), [y_train[i]]))

    print("finish making mnist_train.tfrecord")

    with tf.python_io.TFRecordWriter("mnist_test.tfrecord") as writer:
        for i in range(x_test.shape[0]):
            writer.write(serialize_example(x_test[i].reshape([28*28]), [y_test[i]]))

    print("finish making mnist_test.tfrecord")