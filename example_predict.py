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
    parser.add_argument("--load_model", default=None, type=str, help="Load model from directory.")
    parser.add_argument("--multilabel", default=False, action="store_true", help="Multilabel classification.")
    parser.add_argument("--multilabel_nbest", default=None, type=int, help="Take N best classes from multilabel class prediction (exclusive with --multilabel_threshold).")
    parser.add_argument("--multilabel_threshold", default=None, type=float, help="Threshold for multilabel class prediction (exclusive with --multilabel_nbest).")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args=parser.parse_args()

    # Set threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Read data
    TESTDATA_STR="""
,synsemclass,synsemclass_id,lemma,frame,index,sentence
7,"prozkoumat (v-w4633f1),uvážit (v-w7429f1)","vec00090,vec00149",examine,ev-w1213f1,7,"As a result, he said he will ^ examine the Marcos documents sought by the prosecutors to determine whether turning over the filings is self-incrimination."
8,prozkoumat (v-w4633f1),vec00090,examine,ev-w1213f1,23,"While there have been no reports of similar sudden unexplained deaths among diabetics in the U.S., Dr. Sobel said the FDA plans to ^ examine Dr. Toseland's evidence and is considering its own study here."
12,prozkoumat (v-w4633f1),vec00090,explore,ev-w1249f1,9,"Finally, there is one family movie that quite eloquently ^ explores the depth of human emotion -- only its stars are bears."
"""

    data = pd.read_csv(io.StringIO(TESTDATA_STR))

    # Read target (synsemclass_id) strings to integers encoder/decoder.
    if args.multilabel:
        le = sklearn.preprocessing.MultiLabelBinarizer()
    else:
        le = sklearn.preprocessing.LabelEncoder()
    with open("{}/classes.pickle".format(args.load_model), "rb") as pickle_file:
        le = pickle.load(pickle_file)

    # Load the tokenizer
    with open("{}/args.pickle".format(args.load_model), "rb") as pickle_file:
            model_training_args = pickle.load(pickle_file)
    print("Loading tokenizer {}".format(model_training_args.bert), file=sys.stderr, flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_training_args.bert)
  
    # Create TF dataset as input to NN
    inputs = tokenizer(data["sentence"].tolist())["input_ids"] # drop masks
    inputs = tf.ragged.constant(inputs)

    tf_dataset = tf.data.Dataset.from_tensor_slices((inputs))

    tf_dataset = tf_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=args.batch_size))

    # Instantiate and compile the model
    model = synsemclass_classifier_nn.SynSemClassClassifierNN(multilabel=args.multilabel)
    model.compile(len(le.classes_),
              bert=model_training_args.bert,
              decay=model_training_args.learning_rate_decay,
              dropout=model_training_args.dropout,
              epochs=model_training_args.epochs,
              focal_loss_gamma=model_training_args.focal_loss_gamma,
              learning_rate=model_training_args.learning_rate,
              multilabel_loss=model_training_args.multilabel_loss,
              nbest=model_training_args.multilabel_nbest,
              threshold=model_training_args.multilabel_threshold,
              training_batches=0)
    model.load_checkpoint(args.load_model)

    # Predict classes on development data
    predicted_classes = model.predict(tf_dataset, threshold=args.multilabel_threshold, nbest=args.multilabel_nbest)
    predicted_classes = le.inverse_transform(predicted_classes)
    
    for i, row in data.iterrows():
        if args.multilabel:
            print("\t".join([",".join(predicted_classes[i]), row["sentence"]]))
        else:
            print("\t".join([predicted_classes[i], row["sentence"]]))
