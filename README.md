# Machine Learning for the SynSemClass project

This repository contains code for machine learning for the SynSemClass project.
It also accompanies the paper Extending an Event-type Ontology: Adding Verbs and Classes
using Fine-tuned LLMs Suggestions accepted to the 17th Linguistic Annotation Workshop (LAW-XVII) @ ACL 2023.

The SynSemClass ontology itself is described here:

https://ufal.mff.cuni.cz/synsemclass

Recent versions of data published within the SynSemClass project include:

Urešová, Zdeňka; et al., 2022, SynSemClass 4.0, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11234/1-4746.

## License (of the Code)

Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of Mathematics and Physics, Charles University, Czech Republic.

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Requirements

The software has been developed and tested on Linux. It is possible to run the code on CPU, but it will be much faster on a GPU.

## Installation

1. Clone the repository:

```sh
git clone https://github.com/strakova/synsemclass_ml
```

2. Create the Python virtual environment:

```sh
python3 -m venv venv
venv/bin/pip3 install -r requirements.txt
```

## SynSemClass Model Training

The model can be fine-tuned from scratch with the ``classifier.py`` script.

The hyperparameters used for our experiments are described in the paper
Extending an Event-type Ontology: Adding Verbs and Classes using Fine-tuned LLMs
Suggestions accepted to the 17th Linguistic Annotation Workshop (LAW-XVII) @ ACL
2023.

## SynSemClass Prediction

Predict the mention (lemmas) classes from context (sentences). The script
``example_predict.py`` shows how to load the fine-tuned BERT model
(``--load_model=<path_to_directory>``) for prediction.

## Ranking Lemmas in a Corpus (the paper)

Given a fine-tuned LLM classifier on a (partially) annotated ontology (see
SynSemClass Model Training), the yet
unprocessed lemmas in large raw corpus can be sorted from highest to lowest
scored with ``rank_unprocessed_lemmas_in_corpus.py``.

Lemmas are sorted into frequency buckets and printed to separate TXT files.
Merge the buckets into one file with ``merge_buckets.py``.

## How to Cite

Please cite the publication Extending an Event-type Ontology: Adding Verbs and
Classes using Fine-tuned LLMs Suggestions accepted to the 17th Linguistic
Annotation Workshop (LAW-XVII) @ ACL 2023.

We will add the link and BibTex upon publication.

## Contact

Jana Straková
strakova@ufal.mff.cuni.cz
