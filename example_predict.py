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
Example script to predict SynSem classes for lemmas in sentences.
"""


import datetime
import io
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


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--checkpoint_filename", default="checkpoint.h5", type=str, help="Checkpoint filename.")
    parser.add_argument("--load_model", default=None, type=str, help="Load model from directory.")
    parser.add_argument("--multilabel_nbest", default=None, type=int, help="Take N best classes from multilabel class prediction (exclusive with --multilabel_threshold).")
    parser.add_argument("--multilabel_threshold", default=None, type=float, help="Threshold for multilabel class prediction (exclusive with --multilabel_nbest).")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args=parser.parse_args()

    # Set threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Read data
    TESTDATA_STR="""
,synsemclass,synsemclass_id,lang,lemma,frame,index,sentence
13100,end (ev-w1142f2),vec00113,eng,expire,ev-w1245f1,5,"The offer is scheduled to ^ expire on Nov. 28, unless extended."
13101,end (ev-w1142f2),vec00113,eng,expire,ev-w1245f1,10,"Wasserstein Perella&Co. is the dealer-manager for the offer, which will ^ expire Nov. 29, unless extended ."
13103,end (ev-w1142f2),vec00113,eng,expire,ev-w1245f1,4,The current debt limit ^ expires Oct. 31.
"""

    data = pd.read_csv(io.StringIO(TESTDATA_STR))

    # Load model parameters
    with open("{}/args.pickle".format(args.load_model), "rb") as pickle_file:
            model_training_args = pickle.load(pickle_file)

    # Read target (synsemclass_id) strings to integers encoder/decoder.
    with open("{}/classes.pickle".format(args.load_model), "rb") as pickle_file:
        le = pickle.load(pickle_file)

    # Load the tokenizer
    print("Loading tokenizer {}".format(model_training_args.bert), file=sys.stderr, flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_training_args.bert)

    # Create TF dataset as input to NN
    inputs = tokenizer(data["sentence"].tolist())["input_ids"] # drop masks
    inputs = tf.ragged.constant(inputs)

    tf_dataset = tf.data.Dataset.from_tensor_slices((inputs))

    tf_dataset = tf_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=args.batch_size))

    # Instantiate and compile the model
    model = synsemclass_classifier_nn.SynSemClassClassifierNN(multilabel=model_training_args.multilabel,
                                                              checkpoint_filename=args.checkpoint_filename)
    model.compile(len(le.classes_), model_training_args)
    model.load_checkpoint(args.load_model)

    # Predict classes on development data
    predicted_classes = model.predict(tf_dataset, threshold=args.multilabel_threshold, nbest=args.multilabel_nbest)
    predicted_classes = le.inverse_transform(predicted_classes)

    for i, row in data.iterrows():
        if model_training_args.multilabel:
            print("\t".join([",".join(predicted_classes[i]), row["sentence"]]))
        else:
            print("\t".join([predicted_classes[i], row["sentence"]]))
