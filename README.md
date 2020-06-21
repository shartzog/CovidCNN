# CovidCNN
This repository contains a series of CNNs trained on "imagized" data from the Mexican Covid-19 dataset.

Built Using:
 - python 3.7.6
 - pytorch 1.5.0+cpu
 - torchvision 0.6.0+cpu

v1 is a multiclass example that is basically equivalent to the OOTB CIFAR example.
v2 is my first attempt at a multi label model and does not account for age effectively.
v3 incorporates age as the 'alpha' channel rather than a discreet set of pixels and performs fairly well.  I'm still working to optimize the NN structure and improve performance visualization.

To test, extract the appropritate 7z archive to the directory containing the script and rename the .txt file to .tar.

To train, contact me for details on setting up a DB based on the Mexican Covid data set.

testcovid.png is an example of how an "imagized" covid case looks.
