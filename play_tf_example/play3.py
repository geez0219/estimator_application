
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    n_observations = int(3)

    feature0 = np.random.choice([False, True], n_observations)

    feature1 = np.random.randint(0, 5, n_observations)

    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]

    feature3 = np.random.randn(n_observations)

    # print feature
    print("feature0:{}".format(feature0))
    print("feature1:{}".format(feature1))
    print("feature2:{}".format(feature2))
    print("feature3:{}".format(feature3))


    feature_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))

    def _byte_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def serialize_example(feature0, feature1, feature2, feature3):

        feature = {
            "feature0": _int64_feature(feature0),
            "feature1": _int64_feature(feature1),
            "feature2": _byte_feature(feature2),
            "feature3": _float_feature(feature3)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        return example_proto.SerializeToString()


    def tf_serialize_example(f0, f1, f2, f3):
        tf_string = tf.py_function(
            serialize_example,
            (f0, f1, f2, f3),
            tf.string
        )

        return tf.reshape(tf_string, ())

    serialized_feature_dataset = feature_dataset.map(tf_serialize_example)
    filename = "test.tfrecord"

    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(serialized_feature_dataset)

