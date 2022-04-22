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
    --column_y: CSV column with (output, predicted target) variable Y.
        Expecting a SynSemClass id or name (string).
    --column_x: CSV column with (input) variable X. Expecting a sentence
        (string) with the mention introduced by special token ^.

Example input CSV file (--column_x=synsemclass --column_y=sentence):
synsemclass,sentence
brát (v-w202f1),Možná bychom to měli ^ brát jako kompliment.

Example Usage:
./classifier.py \
    --all_data=lemma_examples.csv \
    --column_x=sentence \
    --column_y=synsemclass_id \
    --dev=lemma_examples_dev.csv \
    --train=lemma_examples_train.csv
"""


import datetime
import os
import pickle
import re
import sys

import sklearn.preprocessing
import pandas as pd
import tensorflow as tf
import transformers

import synsemclass_classifier_nn


if __name__ == "__main__":
    import argparse
   
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_data", default=None, type=str, help="All data.")
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--column_x", default="sentence", type=str, help="CSV column with the (input) variable X. Expecting a sentence (string) with the mention introduced by special token ^.")
    parser.add_argument("--column_y", default="synsemclass_id", type=str, help="CSV column with (output, predicted target) variable Y. Expecting a SynSemClass id or name (string).")
    parser.add_argument("--dev", default=None, type=str, help="Dev data.")
    parser.add_argument("--dev_predictions", default=None, type=str, help="Dev predictions filename.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout.")
    parser.add_argument("--epochs", default=10, type=int, help="Epochs.")
    parser.add_argument("--focal_loss_gamma", default=2.0, type=float, help="Focal loss gamma.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_decay", default=False, action="store_true", help="Learning rate decay.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT transformers model (default: bert-base-multilingual-uncased).")
    parser.add_argument("--load_model", default=None, type=str, help="Load model from directory.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir.")
    parser.add_argument("--multilabel", default=False, action="store_true", help="Multilabel classification.")
    parser.add_argument("--multilabel_loss", default="sigmoid", type=str, help="Multilabel loss (default: sigmoid).")
    parser.add_argument("--multilabel_nbest", default=None, type=int, help="Take N best classes from multilabel class prediction (exclusive with --multilabel_threshold).")
    parser.add_argument("--multilabel_threshold", default=None, type=float, help="Threshold for multilabel class prediction (exclusive with --multilabel_nbest).")
    parser.add_argument("--save_model", default=None, type=str, help="Save model to directory.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train", default=None, type=str, help="Train data.")
    parser.add_argument("--train_size", default=None, type=int, help="Limit train size.")
    parser.add_argument("--dev_size", default=None, type=int, help="Limit dev size.")
    args=parser.parse_args()

    # Process commandline arguments
    if args.multilabel_threshold and args.multilabel_nbest:
        raise ValueError("Arguments --multilabel_threshold and --multilabel_nbest must be used exclusively.")

    logargs = dict(vars(args).items())
    del logargs["dev_predictions"]
    del logargs["load_model"]
    del logargs["save_model"]
    del logargs["train"]
    args.logdir = os.path.join(args.logdir, "{}-{}-{}".format(
                os.path.basename(globals().get("__file__", "notebook")),
                datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(logargs.items())))
            ))

    # Create logdir
    if not os.path.exists(args.logdir):
        print("Making logdir {}".format(args.logdir), file=sys.stderr, flush=True)
        os.mkdir(args.logdir)

    # Read data
    if args.all_data:
        all_data = pd.read_csv(args.all_data)
    if args.train:
        train = pd.read_csv(args.train)
    if args.dev:
        dev = pd.read_csv(args.dev)

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

    print("Loading tokenizer {}".format(args.bert), file=sys.stderr, flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert)

    model = synsemclass_classifier_nn.SynSemClassClassifierNN(threads=args.threads,
                                                              seed=args.seed,
                                                              multilabel=args.multilabel)
    
    def create_dataset(data, size, shuffle=True):
        """Creates TensorFlow dataset."""

        inputs = tokenizer(data[args.column_x].tolist())["input_ids"] # drop masks
        inputs = tf.ragged.constant(inputs)

        # output layer
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

    model.compile(len(le.classes_),
                  bert=args.bert,
                  decay=args.learning_rate_decay,
                  dropout=args.dropout,
                  epochs=args.epochs,
                  focal_loss_gamma=args.focal_loss_gamma,
                  learning_rate=args.learning_rate,
                  multilabel_loss=args.multilabel_loss,
                  nbest=args.multilabel_nbest,
                  threshold=args.multilabel_threshold,
                  training_batches=len(tf_train_dataset) if args.train else 0)

    # Load model or fine-tune
    if args.load_model:
        model.load_checkpoint(args.load_model)
    else:
        model.train(tf_train_dataset, tf_dev_dataset=tf_dev_dataset, epochs=args.epochs, logdir=args.logdir)

    # Save the model
    if args.save_model:
        path = "{}/{}".format(args.logdir, args.save_model)
        model.save_checkpoint(path)
        with open("{}/classes.pickle".format(path), "wb") as pickle_file:
            pickle.dump(le, pickle_file)

    # Predict classes on development data
    if args.dev_predictions:
        print("Predicting dev data {} into file {}".format(args.dev, args.dev_predictions), file=sys.stderr, flush=True)
        
        predicted_classes = model.predict(tf_dev_dataset, threshold=args.multilabel_threshold, nbest=args.multilabel_nbest)
        predicted_classes = le.inverse_transform(predicted_classes)

        with open("{}/{}".format(args.logdir, args.dev_predictions), "w") as fw:
            for i, row in dev.iterrows():
                if args.dev_size and i == args.dev_size: break
                if args.multilabel:
                    print("\t".join([",".join(predicted_classes[i]), row[args.column_y], row[args.column_x]]), file=fw)
                else:
                    print("\t".join([predicted_classes[i], row[args.column_y], row[args.column_x]]), file=fw)
