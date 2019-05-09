import tensorflow as tf
import numpy as np
import pickle


def ResNet50(x):
    def ResNet_block(x, filters, mode):
        if mode == 0:  # the dense block size remain the same
            # first part
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[1, 1], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            # second part
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            # third part
            upstream = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=[1, 1], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)

            # downstream
            downstream = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=[1, 1], padding="same",
                                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        else:  # the dense block size downsample by half
            # first part
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[1, 1], padding="same", strides=[2,2],
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            # second part
            upstream = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)
            upstream = tf.keras.layers.Activation("relu")(upstream)

            # third part
            upstream = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=[1, 1], padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(upstream)
            upstream = tf.keras.layers.BatchNormalization()(upstream)

            # downstream
            downstream = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=[1, 1], padding="same", strides=[2,2],
                                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        output = tf.keras.layers.add([upstream, downstream])
        output = tf.keras.layers.Activation("relu")(output)

        return output

    stack_num = 4
    filter_num = [32, 64, 128, 128]
    depth = [3, 4, 6, 3]

    assert x.get_shape().as_list()[1:]  == [64,64,3]
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    for stack_idx in range(stack_num):
        with tf.name_scope("stack{}".format(stack_idx)):
            for block_idx in range(depth[stack_idx]):
                if stack_idx > 0 and block_idx == 0:
                    mode = 1
                else:
                    mode = 0
                x = ResNet_block(x, filters=filter_num[stack_idx], mode=mode)

    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=200, activation=None)(x)

    return x


def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params["feature_column"])
    net = tf.reshape(net, shape=[-1, 64, 64, 3])
    logits = ResNet50(net)
    assert logits.get_shape().as_list()[1:] == [params["n_classes"]]
    predicted_classes = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction = {
            "class_ids": logits[:, tf.newaxis],
            "probability": tf.nn.softmax(logits),
            "logits": logits
        }

        return tf.estimator.EstimatorSpec(mode, prediction)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name="acc_op")

    tf.summary.scalar("accuracy", accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {"accuracy": accuracy}

        return tf.estimator.EstimatorSpec(mode, metrics)

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])
    train_op = optimizer.minimize(loss,  global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":

    def parse_fn(x):
        feature_dict = {
            "pixel": tf.FixedLenFeature([64*64*3], tf.float32),
            "label": tf.FixedLenFeature([], tf.int64)
        }
        example = tf.parse_single_example(x, feature_dict)
        example["pixel"] = example["pixel"]/255.0
        example = (example, example["label"])

        return example

    # define input function
    def train_input_fn(batch_size):
        dataset = tf.data.TFRecordDataset(["train.tfrecord"])
        # dataset = dataset.map(map_func=parse_fn)
        dataset = dataset.map(map_func=parse_fn, num_parallel_calls=12)
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset


    # def predict_input_fn(features, labels, batch_size):
    #     dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    #     dataset = dataset.batch(batch_size=batch_size)
    #     return dataset
    #
    #
    def eval_input_fn(batch_size):
        dataset = tf.data.TFRecordDataset(["test.tfrecord"])
        # dataset = dataset.map(parse_fn)
        # dataset = dataset.batch(batch_size=batch_size)

        dataset = dataset.map(map_func=parse_fn, num_parallel_calls=12)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset


    # define feature column
    feature_column = [tf.feature_column.numeric_column(key="pixel", shape=(64, 64, 3), dtype=tf.float32)]

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={"feature_column": feature_column,
                "n_classes": 200,
                "lr": 1e-4},
        model_dir="tensorboard")

    tf.logging.set_verbosity(tf.logging.INFO)
    estimator.train(input_fn=lambda: train_input_fn(128),
                    steps=100000)

    estimator.evaluate(input_fn=lambda: eval_input_fn(128),
                       step=None)





