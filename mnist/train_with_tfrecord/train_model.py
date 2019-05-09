import tensorflow as tf
import numpy as np
import cv2


def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params["feature_column"])

    net = tf.keras.layers.Flatten()(net)
    for i in range(params["n_hidden_layers"]):
        net = tf.keras.layers.Dense(units=params["n_neurons"], activation=tf.nn.relu)(net)

    logits = tf.keras.layers.Dense(units=params["n_classes"], activation=None)(net)

    predicted_classes = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction = {
            "class_ids": predicted_classes[:, tf.newaxis],
            "probability": tf.nn.softmax(logits),
            "logits": logits
        }
        return tf.estimator.EstimatorSpec(mode, prediction)

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name="acc_op")

    metrics = {"accuracy": accuracy}
    tf.summary.scalar("accuracy", accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    def parse_fn(x):
        features = {
            "pixel": tf.FixedLenFeature([28*28], tf.float32),
            "label": tf.FixedLenFeature([], tf.int64)
        }

        example = tf.parse_single_example(x, features)
        example = (example, example["label"])

        return example

    def train_input_fn(batch_size):
        dataset = tf.data.TFRecordDataset(["mnist_train.tfrecord"])
        dataset = dataset.map(parse_fn)
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        return dataset

    def eval_input_fn(batch_size):
        dataset = tf.data.TFRecordDataset(["mnist_test.tfrecord"])
        dataset = dataset.map(parse_fn).batch(batch_size)
        return dataset


    # define feature column
    feature_column = [tf.feature_column.numeric_column(key="pixel", shape=(28,28), dtype=tf.float32)]
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={"n_neurons": 128,
                "n_hidden_layers": 3,
                "n_classes": 10,
                "feature_column": feature_column,
                "lr": 1e-4},
        model_dir="tensorboard"
    )
    #
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator.train(input_fn=lambda: train_input_fn(128),
                    steps=1000)

    estimator.evaluate(input_fn=lambda: eval_input_fn(128),
                       steps=None)


