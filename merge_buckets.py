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
Merges buckets with ranked lemmas classified by rank_unprocessed_lemmas_in_corpus.py.

Inputs:
    --buckets: list of bucket files classified by
               rank_unprocessed_lemmas_in_corpus.py, separated by a comma.
    --examples: Optional. If CSV with examples is given with synsemclass_id and
                synsemclass fields, the script additionally adds the class names.
    --output: Output TXT file.
"""


import csv
import sys


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets",
                        default="bucket_2753494_1_freq_to_16.txt,bucket_2753494_2_freq_to_49.txt,bucket_2753494_3_freq_to_122.txt,bucket_2753494_4_freq_to_312.txt,bucket_2753494_5_freq_to_999999.txt",
                        type=str,
                        help="Bucket files.")
    parser.add_argument("--examples",
                        default=None,
                        type=str,
                        help="Examples CSV file to extract class names (optional).")
    parser.add_argument("--langs_priority",
                        type=str,
                        default="ces,eng,spa,deu",
                        help="Preferred order of allowed languages in class names displayed.")
    parser.add_argument("--output", default="all_buckets_2753494.txt", type=str, help="All merged buckets.")
    args=parser.parse_args()

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

    lines = dict()
    for bucket in args.buckets.split(","):
        with open(bucket, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line = line.rstrip()
                cols = line.split("\t")
                for i in range(3, len(cols), 2):
                    if args.examples and cols[i] in classes:
                        cols[i] = "{} / {}".format(cols[i], classes[cols[i]])
                line = "\t".join(cols)
                lines[line] = float(cols[2])

    with open(args.output, "w", encoding="utf-8") as fw:
        for line, _ in sorted(lines.items(), key=lambda x: x[1]):
            print(line, file=fw)
