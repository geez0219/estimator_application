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
                                   prediction=predicted_classes,
                                   name="acc_op")

    tf.summary.scalar("accuracy", accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {"accuracy": accuracy}

        return tf.estimator.EstimatorSpec(mode, metrics)

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])
    train_op = optimizer.minimize(loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    # preprocess training data
    (x_train, y_train), (x_test, y_test) = pickle.load(open("dataset.pkl", "rb"))
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # define input function
    def train_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(x_train.shape[0]).repeat().batch(batch_size)
        return dataset


    def predict_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(batch_size=batch_size)
        return dataset


    def eval_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(batch_size=batch_size)
        return dataset


    # define feature column
    feature_column = [tf.feature_column.numeric_column(key="pixels", shape=(64,64,3), dtype=tf.float32)]

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={"feature_column": feature_column,
                "n_classes": 200,
                "lr": 1e-4},
        model_dir="tensorboard")

    tf.logging.set_verbosity(tf.logging.INFO)
    estimator.train(input_fn=lambda: train_input_fn(x_train, y_train, 128),
                    steps=100000)





