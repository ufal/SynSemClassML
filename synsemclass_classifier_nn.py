#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""SynSemClass TensorFlow Neural Network for Classification."""


import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import transformers


class SynSemClassClassifierNN:
    """SynSemClass TensorFlow neural network for classification."""


    def __init__(self, multilabel=False, checkpoint_filename="checkpoint.h5"):
        """Initializes the network."""

        self._multilabel = multilabel
        self._checkpoint_filename = checkpoint_filename


    def compile(self, output_layer_dim,
                args,
                training_batches=0):
        """Compiles the model.

        Receives Input:
            tf_test_dataset: tf.Dataset with a pair of (1.) tf.RaggedTensors of
            shape [batch_size, sentence_length], containing the tokenizer
            subword ids and (2.) gold labels.
        """

        # 1. Input
        input_token_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, ragged=True, name="input_token_ids")

        # 2. BERT
        bert = transformers.TFAutoModel.from_pretrained(args.bert)
        embeddings = bert(input_token_ids.to_tensor(), attention_mask=tf.sequence_mask(input_token_ids.row_lengths())).last_hidden_state[:,0]

        # 3. Dropout
        dropout_layer = tf.keras.layers.Dropout(args.dropout)(embeddings)

        # 4. Output
        output_layer = tf.keras.layers.Dense(output_layer_dim, activation=args.loss)(dropout_layer)

        self._model = tf.keras.Model(inputs=input_token_ids, outputs=output_layer)

        class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
            """Custom Linear Warmup for learning rate."""

            def __init__(self, warmup_steps, following_schedule):
                self._warmup_steps = warmup_steps
                self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
                self._following = following_schedule

            def __call__(self, step):
                return tf.cond(step < self._warmup_steps,
                               lambda: self._warmup(step),
                               lambda: self._following(step - self._warmup_steps))

        if self._multilabel:
            class CustomF1Score(tfa.metrics.F1Score):
                """Ensures at least one class is predicted before F1Score."""

                def update_state(self, y_true, y_pred, sample_weight=None):
                    _, largest = tf.math.top_k(y_pred, k=args.multilabel_nbest or 1, sorted=False)
                    y_pred += tf.math.reduce_sum(tf.one_hot(largest, output_layer_dim), axis=1)
                    return super().update_state(y_true, y_pred, sample_weight)

            f1score = CustomF1Score(num_classes=output_layer_dim, average="micro", threshold=args.multilabel_threshold or 1.0)

            if args.loss == "sigmoid":
                loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=args.focal_loss_gamma)
            elif args.loss == "softmax":
                loss=tf.keras.losses.CategoricalCrossentropy()
            metrics = f1score
        else:
            if args.loss == "sigmoid":
                # Make our own SparseBinaryFocalCrossEntropy
                loss=lambda y_t, y_p: tf.keras.losses.BinaryFocalCrossentropy(gamma=args.focal_loss_gamma)(tf.one_hot(y_t, output_layer_dim, axis=1), y_p)
            elif args.loss == "softmax":
                loss=tf.keras.losses.SparseCategoricalCrossentropy()
            metrics=tf.keras.metrics.SparseCategoricalAccuracy()

        if args.learning_rate_decay:
            learning_rate_fn = tf.optimizers.schedules.CosineDecay(args.learning_rate, training_batches * (args.epochs - args.warmup_epochs))
        else:
            learning_rate_fn = lambda _: args.learning_rate

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LinearWarmup(training_batches * args.warmup_epochs, learning_rate_fn)),
            loss=loss, metrics=metrics)


    def load_checkpoint(self, dirname):
        """Loads model from directory.

        Loads checkpoint from directory "dirname".

        Receives Input:
            dirname: Path to directory (string).
        """

        print("Loading checkpoint from directory {}".format(dirname), file=sys.stderr, flush=True)
        self._model.load_weights("{}/{}".format(dirname, self._checkpoint_filename))
        print("Model loaded.", file=sys.stderr, flush=True)


    def save_checkpoint(self, dirname):
        """Saves checkpoint to directory.

        Saves checkpoint to directory "dirname".
        Recursively creates directory if "dirname" path does not exist.

        Receives Input:
            dirname: Path to directory (string).
        """

        if not os.path.isdir(dirname):
            print("Creating directory {}".format(dirname), file=sys.stderr, flush=True)
            os.makedirs(dirname)

        print("Saving checkpoint to directory {}".format(dirname), file=sys.stderr, flush=True)
        self._model.save_weights("{}/{}".format(dirname, self._checkpoint_filename))


    def train(self, tf_train_dataset, tf_dev_dataset, epochs=10, logdir=None):
        """Fine-tunes the model."""

        print("Fine-tuning.", file=sys.stderr, flush=True)
        self._model.fit(tf_train_dataset,
              validation_data=tf_dev_dataset,
              epochs=epochs,
              verbose=2,
              callbacks=[tf.keras.callbacks.TensorBoard(logdir)])


    def predict(self, tf_test_dataset, threshold=None, nbest=None):
        """Predicts test data classes.

        Receives Input:
            tf_test_dataset: tf.Dataset with tf.RaggedTensors of shape
            [batch_size, sentence_length], containing the tokenizer subword
            ids.

        Returns Output:
            For multilabel prediction (self._multilabel == True):
                Python 2D array of examples x classes, with 1's for predicted
                classes and 0's otherwise.
                if threshold:
                    Classes with value higher that threshold are predicted. At
                    least one class (maximum predicted value) has always 1.
                if nbest:
                    Classes with nbest highest values are predicted.
                Either threshold or nbest, but not both, must be given for
                multilabel prediction (if self._multilabel == True), otherwise
                ValueError is raised.
                The classes positions can be decoded with
                sklearn.preprocessing.MultiLabelBinarizer().
            For 1-label prediction (self._multilabel == False):
                Python 1D array of examples, each example labeled with exactly
                one integer, the class with maximum probability.
                The cardinals can be decoded with
                sklearn.preprocessing.LabelEncoder().
        """

        if self._multilabel:
            if threshold and nbest:
                raise ValueError("SynSemClassClassifierNN: predict(): Arguments threshold nbest must be used exclusively.")

        predicted_values = self._model.predict(tf_test_dataset)

        if self._multilabel:
            if threshold:
                for i in range(len(predicted_values)):
                    predicted_values[i][np.argmax(predicted_values[i])] = 1
                    for j in range(len(predicted_values[i])):
                        predicted_values[i][j] = 1 if predicted_values[i][j] >= threshold else 0
                return predicted_values
            elif nbest:
                for i in range(len(predicted_values)):
                    ind = np.argpartition(predicted_values[i], -nbest)[-nbest:]
                    for j in range(len(predicted_values[i])):
                        predicted_values[i][j] = 1 if j in ind else 0
                return predicted_values
            else:
                raise ValueError("SynSemClassClassifierNN: predict(): If self_multilabel, either threhold or nbest argument must be specified.")
        else:
            predicted_classes = [0] * len(predicted_values)
            for i, values in enumerate(predicted_values):
                predicted_classes[i] = np.argmax(values)
            return predicted_classes


    def predict_values(self, tf_test_dataset, verbose=1):
        """Predicts test data classes probs/logits.

        Receives Input:
            tf_test_dataset: tf.Dataset with tf.RaggedTensors of shape
            [batch_size, sentence_length], containing the tokenizer subword
            ids.

        Returns Output:
            For multilabel prediction (self._multilabel == True):
                Python 2D array of examples x classes, with values/probs for
                each class (when sigmoid/softmax activation function was used,
                respectively).
                The classes positions can be decoded with
                sklearn.preprocessing.MultiLabelBinarizer().
            1-label prediction (self._multilabel = False)
                Python 2D array of examples x classes, with probs for each class.
                The cardinals can be decoded with
                sklearn.preprocessing.LabelEncoder().

        """

        return self._model.predict(tf_test_dataset, verbose=verbose)
