# Machine Learning for the SynSemClass project

This repository contains code for machine learning for the SynSemClass project.
It also accompanies [Extending an Event-type Ontology: Adding Verbs and Classes using Fine-tuned LLMs Suggestions](https://aclanthology.org/2023.law-1.9/).

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

The hyperparameters used for our experiments are described in [Extending an Event-type Ontology: Adding Verbs and Classes using Fine-tuned LLMs Suggestions](https://aclanthology.org/2023.law-1.9/).

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

Please cite [Extending an Event-type Ontology: Adding Verbs and Classes using Fine-tuned LLMs Suggestions](https://aclanthology.org/2023.law-1.9/):

```
@inproceedings{strakova-etal-2023-extending,
    title = "Extending an Event-type Ontology: Adding Verbs and Classes Using Fine-tuned {LLM}s Suggestions",
    author = "Strakov{\'a}, Jana  and
      Fu{\v{c}}{\'\i}kov{\'a}, Eva  and
      Haji{\v{c}}, Jan  and
      Ure{\v{s}}ov{\'a}, Zde{\v{n}}ka",
    booktitle = "Proceedings of the 17th Linguistic Annotation Workshop (LAW-XVII)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.law-1.9",
    pages = "85--95",
    abstract = "In this project, we have investigated the use of advanced machine learning methods, specifically fine-tuned large language models, for pre-annotating data for a lexical extension task, namely adding descriptive words (verbs) to an existing (but incomplete, as of yet) ontology of event types. Several research questions have been focused on, from the investigation of a possible heuristics to provide at least hints to annotators which verbs to include and which are outside the current version of the ontology, to the possible use of the automatic scores to help the annotators to be more efficient in finding a threshold for identifying verbs that cannot be assigned to any existing class and therefore they are to be used as seeds for a new class. We have also carefully examined the correlation of the automatic scores with the human annotation. While the correlation turned out to be strong, its influence on the annotation proper is modest due to its near linearity, even though the mere fact of such pre-annotation leads to relatively short annotation times.",
}
```

## Contact

Jana Straková
strakova@ufal.mff.cuni.cz
