# CovidCNN
This repository contains a series of CNNs trained on "imagized" data from the Mexican Covid-19 dataset.

Built Using:
 - python 3.7.6
 - pytorch 1.5.0+cpu
 - torchvision 0.6.0+cpu

Modules:
 - network_dictionary_builder.py:
    builds a series of randomized CNNs based on provided test tensor.<br>
    kwargs can be used to apply constraints including a list of optimizers to test for all nets.
    includes training, importing, and exporting functions for entire net dict.
    NOT specific to CovidCNN.  Could be utilized in other applications.
 - network_dictionary_analyzer.py:
    aggregates data for all nets/optimizers in a given network dictionary.
    provides trending functions to analyze and compare randomized nets.
    NOT specific to CovidCNN.  Could be utilized in other applications.
 - utilities.py:
    functions required to support CovidCNN efforts.
    includes class for initializing, storing, and recalling train/test data.
    translates binary string data (e.g. "00100110") + pt age to image and vice versa.

See candidate_cnn_builder.ipynb for an example of how these modules can be implemented.
