"""
1. try to serialize one example and store it in TFRecord file
2. try to serialize one example and disserialize it back
"""


import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    def _byte_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    n_observations = int(1e4)

    feature0 = np.random.choice([False, True], n_observations)

    feature1 = np.random.randint(0, 5, n_observations)

    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]

    feature3 = np.random.randn(n_observations)

    def serialize_example(feature0, feature1, feature2, feature3):

        feature = {
            "feature0": _int64_feature(feature0),
            "feature1": _int64_feature(feature1),
            "feature2": _byte_feature(feature2),
            "feature3": _float_feature(feature3)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        return example_proto.SerializeToString()

    serialized_example = serialize_example(False, 4, b'goat', 0.9876)

    with tf.python_io.TFRecordWriter("test.tfrecord") as writer:
        writer.write(serialized_example)

    example_proto = tf.train.Example.FromString(serialized_example)
    print(example_proto)



    # read the TFRecord file using TFRecordDataset
    def _parse_function(example_proto):
        read_feature = {
            "feature0": tf.FixedLenFeature([], dtype=tf.int64),
            "feature1": tf.FixedLenFeature([], dtype=tf.int64),
            "feature2": tf.FixedLenFeature([], dtype=tf.string),
            "feature3": tf.FixedLenFeature([], dtype=tf.float32)
        }

        parsed_features = tf.parse_single_example(example_proto, read_feature)
        return parsed_features

    filename = ["test.tfrecord"]
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()

    with tf.Session() as sess:
        for i in range(1):
            a = sess.run(el)
            print()



