#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Trains a model to classify mentions given contexts into SynSem classes.

Input: CSV file with fields:
    --column_y: CSV column name which contains target variable Y (e.g.,
                "synsemclass_id").
    --column_x: CSV column name which contains input variable X (e.g,
                "sentence"). Contains sentence with special character "^"
                before mention.

Example input CSV file (--column_x=sentence --column_y=synsemclass_id):
synsemclass_id,sentence
vec00402,Možná bychom to měli ^ brát jako kompliment.

Example Usage:
./classifier.py \
    --all_data=examples.csv \
    --column_x=sentence \
    --column_y=synsemclass_id \
    --dev=examples_dev.csv \
    --train=examples_train.csv
"""


import datetime
import os
import pickle
import re
import sys

import numpy as np
import sklearn.preprocessing
import pandas as pd
import tensorflow as tf
import transformers

import synsemclass_classifier_nn


def predict_and_print(data, tf_dataset, filename, args, size=None):
    """Makes predictions for tf_dataset and prints to filename."""

    if args.print_values:
        predicted_values = model.predict_values(tf_dataset)

        with open("{}/{}".format(args.logdir, filename), "w") as fw:
            for i, row in data.iterrows():
                if size and i == size: break
                output = [row.lemma, row.frame, row.sentence]
                for j in range(len(le.classes_)):
                    output.append(le.classes_[j])
                    output.append(str(predicted_values[i][j]))
                print("\t".join(output), file=fw)

    else:
        predicted_classes = model.predict(tf_dataset, threshold=args.multilabel_threshold, nbest=args.multilabel_nbest)
        predicted_classes = le.inverse_transform(predicted_classes)

        with open("{}/{}".format(args.logdir, filename), "w") as fw:
            for i, row in data.iterrows():
                if size and i == size: break
                if data[args.column_y].isnull().values.any():   # no gold data
                    if args.multilabel:
                        print("\t".join([",".join(predicted_classes[i]), row[args.column_x]]), file=fw)
                    else:
                        print("\t".join([predicted_classes[i], row[args.column_x]]), file=fw)
                else:   # with gold data
                    if args.multilabel:
                        print("\t".join([",".join(predicted_classes[i]), row[args.column_y], row[args.column_x]]), file=fw)
                    else:
                        print("\t".join([predicted_classes[i], row[args.column_y], row[args.column_x]]), file=fw)


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_data", default=None, type=str, help="All data.")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT transformers model (default: bert-base-multilingual-uncased).")
    parser.add_argument("--checkpoint_filename", default="checkpoint.h5", type=str, help="Checkpoint filename.")
    parser.add_argument("--column_x", default="sentence", type=str, help="CSV column with the (input) variable X. Expecting a sentence (string) with the mention introduced by special token ^.")
    parser.add_argument("--column_y", default="synsemclass_id", type=str, help="CSV column with (output, predicted target) variable Y. Expecting a SynSemClass id or name (string).")
    parser.add_argument("--dev", default=None, type=str, help="Dev data.")
    parser.add_argument("--dev_predictions", default=None, type=str, help="Dev predictions filename.")
    parser.add_argument("--dev_size", default=None, type=int, help="Limit dev size.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout.")
    parser.add_argument("--epochs", default=10, type=int, help="Epochs.")
    parser.add_argument("--focal_loss_gamma", default=2.0, type=float, help="Focal loss gamma.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_decay", default=False, action="store_true", help="Learning rate decay.")
    parser.add_argument("--load_model", default=None, type=str, help="Load model from directory.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir.")
    parser.add_argument("--loss", default="sigmoid", type=str, help="Loss (default: sigmoid).")
    parser.add_argument("--multilabel", default=False, action="store_true", help="Multilabel classification.")
    parser.add_argument("--multilabel_nbest", default=None, type=int, help="Take N best classes from multilabel class prediction (exclusive with --multilabel_threshold).")
    parser.add_argument("--multilabel_threshold", default=None, type=float, help="Threshold for multilabel class prediction (exclusive with --multilabel_nbest).")
    parser.add_argument("--print_values", default=False, action="store_true", help="Print predicted values.")
    parser.add_argument("--save_model", default=None, type=str, help="Save model to directory.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--test", default=None, type=str, help="Test data.")
    parser.add_argument("--test_size", default=None, type=int, help="Limit test size.")
    parser.add_argument("--test_predictions", default=None, type=str, help="Test predictions filename.")
    parser.add_argument("--train", default=None, type=str, help="Train data.")
    parser.add_argument("--train_size", default=None, type=int, help="Limit train size.")
    parser.add_argument("--warmup_epochs", default=1, type=int, help="Warmup epochs.")
    args=parser.parse_args()

    # Process commandline arguments
    if args.multilabel_threshold and args.multilabel_nbest:
        raise ValueError("Arguments --multilabel_threshold and --multilabel_nbest must be used exclusively.")

    logargs = dict(vars(args).items())
    del logargs["all_data"]
    del logargs["column_x"]
    del logargs["column_y"]
    del logargs["checkpoint_filename"]
    del logargs["dev"]
    del logargs["dev_predictions"]
    del logargs["load_model"]
    del logargs["save_model"]
    del logargs["test"]
    del logargs["test_predictions"]
    del logargs["train"]
    logargs["bert"] = logargs["bert"].split("/")[-1]  # only name (clashed with directories)
    args.logdir = os.path.join(args.logdir, "{}-{}-{}".format(
                os.path.basename(globals().get("__file__", "notebook")),
                datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(logargs.items())))
            ))

    # Create logdir
    if not os.path.exists(args.logdir):
        print("Making logdir {}".format(args.logdir), file=sys.stderr, flush=True)
        os.mkdir(args.logdir)

    # Set random seed and threads globally
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    ### MODEL TRAINING ARGS ###

    if args.load_model:
        with open("{}/args.pickle".format(args.load_model), "rb") as pickle_file:
                model_training_args = pickle.load(pickle_file)

        # Reuse the trained model arguments for consistency
        args.bert = model_training_args.bert
        args.multilabel = model_training_args.multilabel

    ### DATA ###

    # Read data
    if args.all_data:
        all_data = pd.read_csv(args.all_data)
    if args.train:
        train = pd.read_csv(args.train)
    if args.dev:
        dev = pd.read_csv(args.dev)
    if args.test:
        test = pd.read_csv(args.test)

    # Encode targets (SynSemClass id or name) strings as integers
    if args.multilabel:
        le = sklearn.preprocessing.MultiLabelBinarizer()
    else:
        le = sklearn.preprocessing.LabelEncoder()

    if args.load_model:
        with open("{}/classes.pickle".format(args.load_model), "rb") as pickle_file:
            le = pickle.load(pickle_file)
    else:
        if args.multilabel:
            gold_labels = []
            for i, labels_str in enumerate(all_data[args.column_y]):
                gold_labels.append(labels_str.split(","))
            le.fit(gold_labels)
        else:
            le.fit(all_data[args.column_y])

    ### TOKENIZER ###

    print("Loading tokenizer {}".format(args.bert), file=sys.stderr, flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert)

    ### TF DATASET ###

    def create_dataset(data, size, shuffle=True):
        """Creates TensorFlow dataset as input to SynSemClassClassifierNN."""

        inputs = tokenizer(data[args.column_x].tolist())["input_ids"] # drop masks
        inputs = tf.ragged.constant(inputs)

        # output layer
        if data[args.column_y].isnull().values.any():   # no gold data
            tf_dataset = tf.data.Dataset.from_tensor_slices((inputs))
        else:   # with gold data
            if args.multilabel:
                gold_labels = []
                for i, labels_str in enumerate(data[args.column_y]):
                    gold_labels.append(sorted(labels_str.split(",")))
                outputs = le.transform(gold_labels)
            else:
                outputs = le.transform(data[args.column_y])

            tf_dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

        if size:
            tf_dataset = tf_dataset.take(size)
        if shuffle:
            tf_dataset = tf_dataset.shuffle(10000).apply(
                tf.data.experimental.dense_to_ragged_batch(batch_size=args.batch_size))
        else:
            tf_dataset = tf_dataset.apply(
                tf.data.experimental.dense_to_ragged_batch(batch_size=args.batch_size))
        return tf_dataset

    if args.train:
        tf_train_dataset = create_dataset(train, args.train_size, shuffle=True)
    if args.dev:
        tf_dev_dataset = create_dataset(dev, args.dev_size, shuffle=False)
    if args.test:
        tf_test_dataset = create_dataset(test, args.test_size, shuffle=False)

    ### TRAIN/LOAD MODEL ###

    model = synsemclass_classifier_nn.SynSemClassClassifierNN(multilabel=args.multilabel, checkpoint_filename=args.checkpoint_filename)

    # Load model or fine-tune
    if args.load_model:
        model.compile(len(le.classes_), model_training_args)
        model.load_checkpoint(args.load_model)
    else:
        model.compile(len(le.classes_),
                      args,
                      training_batches=len(tf_train_dataset) if args.train else 0)

        model.train(tf_train_dataset, tf_dev_dataset=tf_dev_dataset, epochs=args.epochs, logdir=args.logdir)

    # Save the model
    if args.save_model:
        path = "{}/{}".format(args.logdir, args.save_model)
        model.save_checkpoint(path)
        with open("{}/classes.pickle".format(path), "wb") as pickle_file:
            pickle.dump(le, pickle_file)
        with open("{}/args.pickle".format(path), "wb") as pickle_file:
            pickle.dump(args, pickle_file)

    ### PREDICTIONS ###

    # Dev data predictions
    if args.dev and args.dev_predictions:
        print("Predicting dev data {} into file {}".format(args.dev, args.dev_predictions), file=sys.stderr, flush=True)
        predict_and_print(dev, tf_dev_dataset, args.dev_predictions, args, size=args.dev_size)

    # Test data predictions
    if args.test and args.test_predictions:
        print("Predicting test data {} into file {}".format(args.test, args.test_predictions), file=sys.stderr, flush=True)
        predict_and_print(test, tf_test_dataset, args.test_predictions, args, size=args.test_size)
