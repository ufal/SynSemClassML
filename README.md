# Machine Learning for the SynSemClass project

This repository contains code for machine learning for the SynSemClass project.
The mother project itself is described here:

https://ufal.mff.cuni.cz/synsemclass

Recent versions of data published within the SynSemClass project include:

Urešová, Zdeňka; et al., 2022, SynSemClass 4.0, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11234/1-4746.

## License (of the Code)

Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of Mathematics and Physics, Charles University, Czech Republic.

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

## Requirements

The software has been developed and tested on Linux. While it is possible to run the code on CPU, it will be much faster on a GPU.

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

## SynSemClass Prediction

1. The fine-tuned BERT model for classification of SynSemClass3.5 is available
   at the [Artificial Intelligence Cluster (AIC)](https://aic.ufal.mff.cuni.cz/).
   Please contact ``strakova@ufal.mff.cuni.cz`` for more information about the model.

2. Predict the mention (lemmas) classes from context (sentences). The script
   ``example_predict.py`` shows how to load the fine-tuned BERT model
   (``--load_model=<path_to_directory>``) for prediction.

## SynSemClass Model Training

The model can be fine-tuned from scratch with the ``classifier.py`` script.
