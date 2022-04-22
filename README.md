# Machine Learning for the SynSemClass project
  
This repository contains code for machine learning for the SynSemClass project.
The mother project itself is described here:

https://ufal.mff.cuni.cz/synsemclass

Recent versions of data published within the SynSemClass project include:

- SynSemClass3.5 with Czech, English and German languages
  (http://hdl.handle.net/11234/1-3750).

Release of SynSemClass4.0 is currently work in progress.

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

1. Download the fine-tuned BERT model (3,7G) for classification:

TODO: upload the model

2. Predict the mention (lemmas) classes from context (sentences). The script
   ``classifier.py`` shows examples how to load the fine-tuned BERT model
   (``--load_model=<path_to_directory>``) and how to use the
   ``SynSemClassClassifierNN`` class imported from
   ``synsemclass_classifier_nn``.
