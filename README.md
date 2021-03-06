# CovidCNN
This repository contains a series of CNNs trained on "imagized" data from the Mexican Covid-19 dataset.

Built Using:
 - python 3.7.6
 - pytorch 1.5.0+cpu
 - torchvision 0.6.0+cpu

Modules:
 - network_dictionary_builder.py:<br>
    builds a series of randomized CNNs based on provided test tensor.<br>
    kwargs can be used to apply constraints including a list of optimizers to test for all nets.<br>
    includes training, importing, and exporting functions for entire net dict.<br>
    NOT specific to CovidCNN.  Could be utilized in other applications.<br>
 - network_dictionary_analyzer.py:<br>
    aggregates data for all nets/optimizers in a given network dictionary.<br>
    provides trending functions to analyze and compare randomized nets.<br>
    NOT specific to CovidCNN.  Could be utilized in other applications.<br>
 - utilities.py:<br>
    functions required to support CovidCNN efforts.<br>
    includes class for initializing, storing, and recalling train/test data.<br>
    translates binary string data (e.g. "00100110") + pt age to image and vice versa.<br>
<br>
See candidate_cnn_builder.ipynb for an example of how these modules can be implemented.
