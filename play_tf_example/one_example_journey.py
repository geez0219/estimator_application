import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    # create dataset in numpy
    n_observations = 1
    feature0 = np.random.choice([False, True], n_observations)
    feature1 = np.random.randint(0, 5, n_observations)
    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]
    feature3 = np.random.randn(n_observations)

    # transform numpy data into Example (direct transform)

    example = tf.train.Example(features=tf.train.Features(feature={
        "feature0": tf.train.Feature(int64_list=tf.train.Int64List(value=feature0)),
        "feature1": tf.train.Feature(int64_list=tf.train.Int64List(value=feature1)),
        "feature2": tf.train.Feature(bytes_list=tf.train.BytesList(value=feature2)),
        "feature3": tf.train.Feature(float_list=tf.train.FloatList(value=feature3))
    }))

    # serialized Example into binary string
    example_proto = example.SerializeToString()
    example_back = tf.train.Example.FromString(example_proto)


    # transform numpy data into Example (define transform function)
    """
    def serialize_example(feature0, feature1, feature2, feature3):
        feature = {
            "feature0": tf.train.Feature(int64_list=tf.train.Int64List(value=feature0)),
            "feature1": tf.train.Feature(int64_list=tf.train.Int64List(value=feature1)),
            "feature2": tf.train.Feature(bytes_list=tf.train.BytesList(value=feature2)),
            "feature3": tf.train.Feature(float_list=tf.train.FloatList(value=feature3))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    example_proto = serialize_example(feature0, feature1, feature2, feature3)
    """
    # save the binary string to TFRecord
    with tf.python_io.TFRecordWriter("example.tfrecord") as writer:
        writer.write(example_proto)


    # read the TFRecord file into Dataset
    dataset = tf.data.TFRecordDataset(["example.tfrecord"])

    # define parsing function
    def parse_fn(x):
        features = {
            # the [] means only one element is in the feature and output is dtype
            # the [1] also means one element, but the output is ndarray([dtype])
            # the [n] means n elements, and the output is ndarray([dtype, dtype, dtype,...])

            "feature0": tf.FixedLenFeature([], dtype=tf.int64),
            "feature1": tf.FixedLenFeature([], dtype=tf.int64),
            "feature2": tf.FixedLenFeature([], dtype=tf.string),
            "feature3": tf.FixedLenFeature([], dtype=tf.float32)
        }
        return tf.parse_single_example(x, features)

    # map the parsing function to Dataset
    dataset = dataset.map(parse_fn)
    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()

    with tf.Session() as sess:
        example_feature_dict = sess.run(el)

    print()
    # test the all element of that example are the same

    print("feature0:{}, feature0_back:{}".format(feature0, example_feature_dict["feature0"]))
    print("feature1:{}, feature1_back:{}".format(feature1, example_feature_dict["feature1"]))
    print("feature2:{}, feature2_back:{}".format(feature2, example_feature_dict["feature2"]))
    print("feature3:{}, feature3_back:{}".format(feature3, example_feature_dict["feature3"]))

    assert float(example_feature_dict["feature0"]) == float(feature0[0])
    assert float(example_feature_dict["feature1"]) == float(feature1[0])
    assert str(example_feature_dict["feature2"]) == str(feature2[0])
    assert float(example_feature_dict["feature3"]) == float(feature3[0])