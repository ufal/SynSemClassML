#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2023 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Given a fine-tuned LLM classifier on a (partially) annotated ontology, sorts
yet unprocessed lemmas in large raw corpus from highest to lowest scored and
prints top k scored classes for each lemma.

Lemmas are sorted into frequency buckets and printed to separate TXT files.
Merge the buckets into one file with merge_buckets.py.

Input:
    --lemmas: A CSV file with one column, a lemma per line (really just a TXT
              file with lemmas).
    --corpus_lemmas: A TXT file with tokenized sentences with lemmas, a sentence per line. Example:

za minulý měsíc být s on mluvit snad šestkrát a k svůj zklamání být shledat , že já mít mnoho co říci .
můj první dojem , že být osoba nějak určitě významný , se proto postupně vytratit a on se prostě stát majitel přepychový zájezdní hotel v sousedství .
a potom být zažít ten jízda , jenž já přivést do rozpaky .

    --corpus_forms: A TXT file with tokenized sentences with surface forms, a sentence per line. Example:

Za minulý měsíc jsem s ním mluvil snad šestkrát a k svému zklamání jsem shledal , že mi nemá mnoho co říci .
Můj první dojem , že je osobou nějak neurčitě významnou , se proto postupně vytratil a on se prostě stal majitelem přepychového zájezdního hotelu v sousedství .
A potom jsem zažil tu jízdu , jež mě přivedla do rozpaků .

Example Usage:
./rank_unprocessed_lemmas_in_cnk.py \
    --lemmas=pdt_lemmas_nezarazene.csv \
    --corpus_lemmas=syn_v4.lemmas
    --corpus_forms=syn_v4.forms
"""


import pickle
import sys

import numpy as np
import tensorflow as tf
import transformers

import synsemclass_classifier_nn


# A token that marks the position of the lemma of interest
SPECIAL_TOKEN = "^"

# Lemmas are sorted into buckets by frequency and printed to separate TXT
# files. We were hoping to prioritize somehow by frequency but we ended up just
# using one merged file of all lemmas. Merge the buckets into one file with
# merge_buckets.py.
INFINITY = 999999
#BUCKET_LIMITS = [3, 8, 17, 40, INFINITY]    # for 275349 sentences
BUCKET_LIMITS = [16, 49, 122, 312, INFINITY]    # for 2753494 sentences


def score_sentences(model, sentences):
    """Predicts classifier class scores for a batch of sentences."""

    # Create TF dataset as input to NN
    inputs = tokenizer(batch_sentences)["input_ids"] # drop masks
    inputs = tf.ragged.constant(inputs)
    tf_dataset = tf.data.Dataset.from_tensor_slices((inputs))
    tf_dataset = tf_dataset.apply(
    tf.data.experimental.dense_to_ragged_batch(batch_size=args.batch_size))

    # Predict classes
    return model.predict_values(tf_dataset)


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--corpus_lemmas", default="syn_v4.lemmas", type=str, help="Corpus sentences with lemmas, one sentence per line.")
    parser.add_argument("--corpus_forms", default="syn_v4.forms", type=str, help="Corpus sentences with forms, one sentence per line.")
    parser.add_argument("--lemmas", default="pdt_lemmas_nezarazene.csv", type=str, help="Unprocessed lemmas to count in the corpus.")
    parser.add_argument("--load_model", default=None, type=str, help="Load model from directory.")
    parser.add_argument("--max_lines", default=10, type=int, help="Maximum corpus lines to process.")
    parser.add_argument("--output", default="bucket", type=str, help="Output file template.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--k", default=1, type=int, help="Top k classes.")
    args=parser.parse_args()

    # Set threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load model
    print("Loading model from \'{}\'".format(args.load_model), file=sys.stderr, flush=True)
    with open("{}/args.pickle".format(args.load_model), "rb") as pickle_file:
        model_training_args = pickle.load(pickle_file)
    with open("{}/classes.pickle".format(args.load_model), "rb") as pickle_file:
        le = pickle.load(pickle_file)
    model = synsemclass_classifier_nn.SynSemClassClassifierNN()
    model.compile(len(le.classes_), model_training_args)
    model.load_checkpoint(args.load_model)

    # Load the tokenizer
    print("Loading tokenizer {}".format(model_training_args.bert), file=sys.stderr, flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_training_args.bert)

    # Read list of unprocessed lemmas to classify
    lemmas = dict()
    with open(args.lemmas, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = line.rstrip()
            lemmas[line] = 0

    # Walk through the corpus sentences and classify unprocessed lemmas
    nlines = 0
    batch_lemmas, batch_sentences = [], []
    nbatches = 0
    lemma_class_avgs = dict()
    # Initialize lemma_class_avgs with zero numpy arrays of size le.classes_
    for lemma in lemmas:
        lemma_class_avgs[lemma] = np.zeros(len(le.classes_))
    with open(args.corpus_lemmas, "r", encoding="utf-8") as fr_lemmas:
        with open(args.corpus_forms, "r", encoding="utf-8") as fr_forms:
            line_lemmas = fr_lemmas.readline()
            line_forms = fr_forms.readline()
            nlines += 1
            while line_lemmas:

                if nlines % 1000 == 0:
                    print("Lines counted: {}".format(nlines), file=sys.stderr, flush=True)

                if nlines > args.max_lines:
                    print("Exceeded required maximum number of lines args.max_lines={}, finishing.".format(args.max_lines), file=sys.stderr)
                    break

                line_lemmas = line_lemmas.rstrip()
                line_forms = line_forms.rstrip()
                for lemma in lemmas:
                    lemma_tokens = lemma.split(" ")
                    tokens = line_lemmas.split(" ")
                    forms = line_forms.split(" ")

                    # All lemma tokens must be found in sentence
                    all_found = True
                    lemma_index = None
                    for i, lemma_token in enumerate(lemma_tokens):
                        found = False
                        for j, token in enumerate(tokens):
                            if lemma_token == token:
                                if i == 0:
                                    lemma_index = j
                                found = True
                                break
                        if not found:
                            all_found = False
                            break
                    if all_found:
                        lemmas[lemma] += 1
                        forms.insert(lemma_index, SPECIAL_TOKEN)
                        sentence = " ".join(forms)

                        # Add to batch
                        if len(batch_lemmas) == args.batch_size:
                            scores = score_sentences(model, batch_sentences)
                            for i, batch_lemma in enumerate(batch_lemmas):
                                lemma_class_avgs[batch_lemma] += scores[i]

                            nbatches += 1
                            if nbatches % 10 == 0:
                                print("Batches of size {} classified: {}".format(args.batch_size, nbatches), file=sys.stderr, flush=True)

                            # Clear batch
                            batch_lemmas, batch_sentences = [], []

                        batch_lemmas.append(lemma)
                        batch_sentences.append(sentence)
                line_lemmas = fr_lemmas.readline()
                line_forms = fr_forms.readline()
                nlines += 1

    # Flush last batch
    if batch_sentences:
        scores = score_sentences(model, batch_sentences)
        for i, batch_lemma in enumerate(batch_lemmas):
            lemma_class_avgs[batch_lemma] += scores[i]

    # Compute averages by dividing lemma_class_avgs by lemmas (counts)
    for lemma in lemma_class_avgs:
        if lemmas[lemma] != 0:
            lemma_class_avgs[lemma] /= lemmas[lemma]

    # Compute indices of top k highest scores classes for each lemma.
    top_k = dict()
    for lemma in lemma_class_avgs:
        top_k[lemma] = np.partition(lemma_class_avgs[lemma], -args.k)[-args.k:]

    # Sort lemmas into buckets by frequency
    buckets = [dict() for _ in range(len(BUCKET_LIMITS))]
    for lemma in lemmas:
        if lemmas[lemma] == 0:
            continue

        for i in range(len(BUCKET_LIMITS)):
            if lemmas[lemma] <= BUCKET_LIMITS[i]:
                buckets[i][lemma] = np.average(top_k[lemma])
                highest_class_index = np.argmax(lemma_class_avgs[lemma])
                break

    # In each bucket, sort lemmas from least scored to highest scored.
    for i, bucket in enumerate(buckets):
        with open("{}_{}_freq_to_{}.txt".format(args.output, i+1, BUCKET_LIMITS[i]), "w", encoding="utf-8") as fw:
            for lemma, avg in sorted(buckets[i].items(), key=lambda x: x[1]):
                highest_classes = []
                sorted_lemma_class_avgs = np.argsort(lemma_class_avgs[lemma])[::-1]
                for index in sorted_lemma_class_avgs[:10]:
                    highest_classes.append(le.classes_[index])
                    highest_classes.append("{:.5f}".format(lemma_class_avgs[lemma][index]))
                print("{}\t{}\t{:.5f}\t{}".format(lemma, lemmas[lemma], avg, "\t".join(highest_classes)), file=fw)
