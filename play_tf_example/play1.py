"""
have a taste of object like FloatList, Feature, Features, Example

"""


import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    def _byte_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    float_list1 = tf.train.FloatList(value=[0.1, 0.2, 0.3, 0.4])

    float_list2 = tf.train.FloatList(value=[0.1, 0.2])

    feature1 = tf.train.Feature(float_list=float_list1)
    feature2 = tf.train.Feature(float_list=float_list2)

    features = tf.train.Features(feature={"feature1": feature1,
                                          "feature2": feature2})

    example = tf.train.Example(features=features)

    # test whether can transform ndarray into FloatList
    float_list3 = tf.train.FloatList(value=np.array([[1, 2, 3], [4, 5, 6]]))
    print()
