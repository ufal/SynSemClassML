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


import csv
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
import transformers

import synsemclass_classifier_nn


# A token that marks the position of the lemma of interest
SPECIAL_TOKEN = "^"


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
    parser.add_argument("--examples", default=None, type=str, help="Examples CSV file to extract SynSemClass class names (optional).")
    parser.add_argument("--langs_priority", default="ces,eng,spa,deu", type=str, help="Preferred order of allowed languages in SynSemClass class names displayed.")
    parser.add_argument("--load_model", default=None, type=str, help="Load model from directory.")
    parser.add_argument("--max_hours", default=None, type=int, help="Maximum hours to process for.")
    parser.add_argument("--max_lines", default=None, type=int, help="Maximum corpus lines to process.")
    parser.add_argument("--max_frequency", default=5000, type=int, help="Maximum number of lemma mentions to consider.")
    parser.add_argument("--output", default="lemma_suggestions.txt", type=str, help="Output file template.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--k", default=1, type=int, help="Top k classes.")
    args=parser.parse_args()

    # Process arguments
    if args.max_hours == None and args.max_lines == None:
        raise ValueError("Either --max_hours or --max_lines must be set.")

    if args.max_hours:
        max_runtime = args.max_hours * 60 * 60  # time in seconds
        start_time = time.time()

    # Set threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Read SynSemClass classes names
    langs_priority_dict = dict()
    for i, lang in enumerate(args.langs_priority.split(",")):
        langs_priority_dict[lang] = i

    if args.examples:
        classes = dict()
        classes_langs = dict()
        with open(args.examples, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lang = row["lang"]
                if lang not in langs_priority_dict:
                    continue

                class_id = row["synsemclass_id"]
                if class_id not in classes or langs_priority_dict[lang] < langs_priority_dict[classes_langs[class_id]]:
                    classes[class_id] = row["synsemclass"]
                    classes_langs[class_id] = lang

    # Load model
    print("Loading model from \'{}\'".format(args.load_model), flush=True)
    with open("{}/args.pickle".format(args.load_model), "rb") as pickle_file:
        model_training_args = pickle.load(pickle_file)
    with open("{}/classes.pickle".format(args.load_model), "rb") as pickle_file:
        le = pickle.load(pickle_file)
    model = synsemclass_classifier_nn.SynSemClassClassifierNN()
    model.compile(len(le.classes_), model_training_args)
    model.load_checkpoint(args.load_model)

    # Load the tokenizer
    print("Loading tokenizer {}".format(model_training_args.bert), flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_training_args.bert)

    # Read list of unprocessed lemmas to classify
    lemma_counts = dict()
    with open(args.lemmas, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = line.rstrip()

            # Lemma is in the first column, ignore any other columns
            lemma = line.split("\t")[0]

            if lemma.find("_") != -1:
                old_lemma = lemma
                lemma = lemma.replace("_", " ")
                print("Replaced underscored in lemma from lemma \"{}\" to lemma \"{}\"".format(old_lemma, lemma))

            lemma_counts[lemma] = 0

    # Walk through the corpus sentences and classify unprocessed lemmas
    nlines = 0
    batch_lemmas, batch_sentences = [], []
    nbatches = 0
    lemma_class_avgs = dict()
    reached_max_frequency = dict()

    # Initialize lemma_class_avgs with zero numpy arrays of size le.classes_
    for lemma in lemma_counts:
        lemma_class_avgs[lemma] = np.zeros(len(le.classes_))

    with open(args.corpus_lemmas, "r", encoding="utf-8") as fr_lemmas:
        with open(args.corpus_forms, "r", encoding="utf-8") as fr_forms:
            line_lemmas = fr_lemmas.readline()
            line_forms = fr_forms.readline()
            nlines += 1
            while line_lemmas:

                if nlines % 1000 == 0:
                    print("Lines counted: {}".format(nlines), flush=True)

                # Exit conditions
                if args.max_lines and nlines > args.max_lines:
                    print("Maximum number of lines --max_lines={} reached, finishing".format(args.max_lines))
                    break
                elif args.max_hours and time.time() - start_time > max_runtime:
                    print("Maximum number of hours --max_hours={} reached, finishing".format(args.max_hours))
                    break

                line_lemmas = line_lemmas.rstrip()
                line_forms = line_forms.rstrip()
                for lemma in lemma_counts:
                    # Stop classifying too frequent lemmas
                    if lemma_counts[lemma] >= args.max_frequency:
                        if lemma not in reached_max_frequency:
                            print("Lemma \"{}\" reached maximum frequency {}, stopping classification for this lemma".format(lemma, args.max_frequency))
                            reached_max_frequency[lemma] = 1
                        continue

                    lemma_tokens = lemma.split(" ")
                    tokens = line_lemmas.split(" ")
                    forms = line_forms.split(" ")

                    # All lemma tokens must be found in sentence
                    all_found = True
                    lemma_index = None
                    for i, lemma_token in enumerate(lemma_tokens):
                        # UDPipe tokenizes "si" as "se" in the SYN v4 corpus:
                        if lemma_token == "si":
                            lemma_token = "se"

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
                        lemma_counts[lemma] += 1
                        forms.insert(lemma_index, SPECIAL_TOKEN)
                        sentence = " ".join(forms)

                        # Add to batch
                        if len(batch_lemmas) == args.batch_size:
                            scores = score_sentences(model, batch_sentences)
                            for i, batch_lemma in enumerate(batch_lemmas):
                                lemma_class_avgs[batch_lemma] += scores[i]

                            nbatches += 1
                            if nbatches % 10 == 0:
                                print("Batches of size {} classified: {}".format(args.batch_size, nbatches), flush=True)

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

    # Compute averages by dividing lemma_class_avgs by lemma counts.
    for lemma in lemma_class_avgs:
        if lemma_counts[lemma] != 0:
            lemma_class_avgs[lemma] /= lemma_counts[lemma]

    # Compute averages of top k highest scored classes for each lemma.
    top_k = dict()
    for lemma in lemma_class_avgs:
        if lemma_counts[lemma] != 0:
            top_k[lemma] = np.average(np.partition(lemma_class_avgs[lemma], -args.k)[-args.k:])

    # Sort lemmas from least scored to highest scored in their top k classes.
    with open(args.output, "w", encoding="utf-8") as fw:
        for lemma, avg in sorted(top_k.items(), key=lambda x: x[1]):
            highest_classes_to_print = []
            sorted_lemma_class_avgs = np.argsort(lemma_class_avgs[lemma])[::-1]
            for index in sorted_lemma_class_avgs[:10]:
                synsemclass = le.classes_[index]
                if args.examples and synsemclass in classes:
                    synsemclass = "{} / {}".format(synsemclass, classes[synsemclass])
                highest_classes_to_print.append(synsemclass)
                highest_classes_to_print.append("{:.5f}".format(lemma_class_avgs[lemma][index]))
            print("{}\t{}\t{:.5f}\t{}".format(lemma, lemma_counts[lemma], avg, "\t".join(highest_classes_to_print)), file=fw)

    # Print some stats
    nlemmas = len(lemma_counts)
    found = len(top_k)
    not_found = nlemmas - found
    print("Processed {} lines of the corpus".format(nlines))
    print("Lemmas found in the corpus:\t{} / {} ({:.2f}\%)".format(found, nlemmas, found * 100 / nlemmas))
    print("Lemmas not found in the corpus:\t{} / {} ({:.2f}\%)".format(not_found, nlemmas, not_found * 100 / nlemmas))
