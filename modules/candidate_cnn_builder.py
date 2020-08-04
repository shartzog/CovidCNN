"""
build CNNs from COVID-19 patient data
"""
from __future__ import division, print_function

import math
import os.path
from itertools import permutations
from random import random, randrange
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from tqdm import tqdm
import pyodbc

from network_dictionary_builder import NetDictionary, Net

#use CUDA if available
if torch.cuda.is_available():
    torch.cuda.set_device(0)
else:
    print('**** CUDA not available - continuing with CPU ****')

plt.ion()   # set matplotlib to interactive mode

#container for custom errors
class CustomError(Exception):
    """
    Dummy container for raising errors.
    """

# global constants
NET_PATH = './networks.tar'
DATA_PATH = './datasets.tar'
NETWORK_DEPTH = 4
BATCH_SIZE = 16
LABELS = ('Hospitalized', 'Intubated', 'Deceased', 'Pneumonia')
COLUMNS = ('Male', 'Pregnant', 'Diabetes', 'Asthma', 'Immunocompromised',
           'Hypertension', 'Other Disease', 'Cardiovascular Disease', 'Obesity', 'Kidney Disease',
           'Tobacco Use', 'COPD')
LABEL_COUNT = len(LABELS)
COLUMN_COUNT = len(COLUMNS)
VALIDATION_RATIO = 0.4    #approximate percentage of dataset reserved for validation

#global variable initialization functions
def get_image_dimensions():
    """
    calculates image dimensions from global constants
    inputs (global):
        NETWORK_DEPTH
        COLUMN_COUNT
    """
    feature_count = 1.0
    #using NETWORK_DEPTH - 1 to account for 'age' of patient in final dimension
    for i in range(NETWORK_DEPTH - 1):
        #model will perform NETWORK_DEPTH - 1 permutations for each column
        feature_count *= (COLUMN_COUNT - i)
    image_height_width = math.ceil(feature_count**0.5)
    normalization_tuple = list()
    for i in range(NETWORK_DEPTH):
        normalization_tuple.extend([0.5])
    print('total features: %i; image size: %i x %i x %i'
          % (feature_count, image_height_width, image_height_width, NETWORK_DEPTH))
    return image_height_width, tuple(normalization_tuple)


# global variables
IMAGE_HEIGHT_WIDTH, NORMALIZATION_TUPLE = get_image_dimensions() #constants initialized by function
LOSS_RECORDING_RATE = 10
train_data = list()
validation_images = list()
validation_labels = list()
validation_indices = list()
VALIDATION_DATA_INITIALIZED = os.path.exists(DATA_PATH)
network_dictionary = dict()
loss_dictionary = dict()

#global functions
def create_image_tensor(row):
    """
    Converts binary column data and patient age into an image tensor.
    INPUT:  row:    Expected format is [bindata,age] where bindata is a string of 1s and 0s
                    supplied by the DB to indicate patient conditions
    OUTPUT: Normalized image tensor
    """
    if row:
        icol = 0
        irow = 0
        if len(list(row[0])) != COLUMN_COUNT:
            raise CustomError('Invalid Input Size: Expected %i, Received %i'
                              % (COLUMN_COUNT, len(list(row[0]))))
        if len(row) < 2:
            raise CustomError('Invalid Data: Expected row format is [bindata,age]')
        image_data = np.zeros((IMAGE_HEIGHT_WIDTH, IMAGE_HEIGHT_WIDTH, NETWORK_DEPTH),
                              dtype=np.float32)
        for pixel in permutations(list(row[0]), NETWORK_DEPTH-1):
            this_pixel = list()
            for channel in pixel:
                this_pixel.extend([float(channel)])
            this_pixel.extend([float(row[1])/100.])
            image_data[icol, irow, :] = this_pixel
            irow += 1
            if irow > IMAGE_HEIGHT_WIDTH - 1:
                icol += 1
                irow = 0
        return TF.normalize(TF.to_tensor(image_data), NORMALIZATION_TUPLE, NORMALIZATION_TUPLE)
    else:
        raise CustomError('Invalid Data: No row data supplied.')

def next_row():
    """
    Generator function for retrieving patient data.
    GLOBAL INTERACTION:
        Populates validation_* global variables using a random number method.  NEEDS IMPROVEMENT.
    OUTPUT: tuple of form (tensor of [BATCH_SIZE] images, tensor of [BATCH_SIZE] labels)
    """
    conn = pyodbc.connect('DSN=covid;UID=seh;PWD=Welcome2020!;')
    crsr = conn.cursor()
    crsr.execute("{CALL getpydatav2}")
    rowcnt = 0
    imgs = list()
    labels = list()
    row = [0, 1, 2]
    idx = 0
    while row:
        while rowcnt < BATCH_SIZE:
            if VALIDATION_DATA_INITIALIZED == 1 and idx in validation_indices:
                crsr.skip(BATCH_SIZE)
                row = [0, 1, 2]
                break
            else:
                try:
                    row = crsr.fetchone()
                except:
                    conn = pyodbc.connect('DSN=covid;UID=seh;PWD=Welcome2020!;')
                    crsr = conn.cursor()
                    crsr.execute("{CALL getpydatav2}")
                    crsr.skip(idx*BATCH_SIZE + rowcnt)
                    row = crsr.fetchone()
                rowcnt += 1
                if row:
                    imgtensor = create_image_tensor(row)
                    labels.append(np.array(list(row[2]), dtype=np.float32))
                    imgs.append(imgtensor)
                else:
                    break
        if row:
            if VALIDATION_DATA_INITIALIZED == 0:
                if random() < VALIDATION_RATIO:
                    validation_images.append(torch.stack(imgs))
                    validation_labels.append(torch.tensor(labels))
                    validation_indices.append(idx)
                else:
                    yield torch.stack(imgs), torch.tensor(labels)
            elif idx not in validation_indices:
                yield torch.stack(imgs), torch.tensor(labels)
            rowcnt = 0
            idx += 1
            imgs = list()
            labels = list()
        else:
            break

def gen_test_data(start_index=0):
    """
    Generator function for supplying validation data beginning with a specific index.
    INPUT:  start_index: position within validation_images/validation_labels from which
                         iteration should begin.
    OUTPUT: generator function yeilding a tuple of form ([BATCH_SIZE] images, [BATCH_SIZE] labels)
    """
    for i, labels in enumerate(validation_labels[start_index:], start_index):
        yield validation_images[i], labels

def gettestdata(idx=0):
    """
    MARKED FOR REMOVAL: use 'validation_images[idx], validation_labels[idx]' instead
    """
    return validation_images[idx], validation_labels[idx]

def create_fake_data(age=None, conditions=None):
    """
    Creates a "fake" patient for use in model testing.  Patient age and condition data may be
    specified, provided as a range, or left blank if a random patient is desired.
    INPUTS:
        age:    if an integer, the value is used explicitly.
                if a tuple of integers, a random age between the two values is assigned.
                if None, a random age between 15 and 90 is assigned.
        conditions:
                if a list or string, assign all condtions in the list or string.
                if integer, assign a random set of conditions but do not apply more than this value.
                if None, assign a random set of conditions.
                Integer version is biased to early conditions in COLUMNS.  NEEDS IMPROVEMENT.
    """
    if age is None:
        age = randrange(15, 90)
    elif isinstance(age, tuple): age = randrange(age[0], age[1])
    bindata = ''
    if isinstance(conditions, list) or isinstance(conditions, str):
        for col in COLUMNS:
            bindata += ('1' if col in conditions else '0')
        return create_image_tensor([bindata, age]).unsqueeze(0)
    total_conditions = 0
    if isinstance(conditions, int) and conditions < COLUMN_COUNT:
        total_conditions = conditions
    else:
        total_conditions = math.floor(COLUMN_COUNT*random())
    condition_count = 0
    for i,col in enumerate(COLUMNS):
        if condition_count < total_conditions:
            if random() >= 0.5:
                if i == 1 and (age < 12 or age > 55 or bindata[0] == '1'):
                    bindata += '0'
                else:
                    bindata += '1'
                    condition_count += 1
            else:
                bindata += '0'
        else:
            bindata += '0'
    return create_image_tensor([bindata, age]).unsqueeze(0)

def create_datasets():
    """
    build global datasets
    """
    global train_data, validation_images, validation_indices, validation_labels
    if not VALIDATION_DATA_INITIALIZED:
        rowgen = next_row()
        pbar = tqdm(enumerate(rowgen), total=1200)
        for i, data in pbar:
            train_data.append(data)
        torch.save({'train_data' : train_data,
                    'validation_images' : validation_images,
                    'validation_indices': validation_indices,
                    'validation_labels': validation_labels,
                    }, DATA_PATH)
    else:
        datasets = torch.load(DATA_PATH)
        train_data = datasets['train_data']
        validation_images = datasets['validation_images']
        validation_indices = datasets['validation_indices']
        validation_labels = datasets['validation_labels']
    return True

#main functions
fake_data_list = []
for i in range(BATCH_SIZE):
    fake_data_list.append(create_fake_data(40,["Diabetes"]).squeeze(0))
testtensor = torch.stack(fake_data_list)
network_dictionary = NetDictionary(3, testtensor, LABEL_COUNT, NET_PATH)

create_datasets()

#train a series of layer configurations and record loss data
for k, d in network_dictionary.items():
    net = d['net']
    net.cuda()
    net.train()
    criterion = d['criterion']
    optimizer = d['optimizer'] 
    tlosslist = []
    vlosslist = []
    last_loss = 0.0
    lasttestidx = 0
    running_loss = 0.0
    testcnt = 0
    train_cnt = len(train_data)
    validation_cnt = len(validation_images)
    if not network_dictionary.init_from_file:
        pbar = tqdm(enumerate(train_data), total=train_cnt)
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % LOSS_RECORDING_RATE == LOSS_RECORDING_RATE - 1:
                tlosslist.append((running_loss-last_loss)/LOSS_RECORDING_RATE)
                pbar.set_description(desc='net name: %s; loss: %.3f' % (k, running_loss/(i+1)))
                pbar.update()
                last_loss = running_loss
    last_loss = 0.0
    valid_loss = 0.0
    net.eval()
    with torch.no_grad():
        randomstartindex = 0#math.floor(len(testimages)*random())
        genfunc = gen_test_data(randomstartindex)
        pbar = tqdm(enumerate(genfunc), total=validation_cnt)
        for j, (tin, tlab) in pbar:
            tin, tlab = tin.cuda(), tlab.cuda()
            outputs = net(tin)
            loss = criterion(outputs, tlab)
            valid_loss += loss.item()
            if j % LOSS_RECORDING_RATE == LOSS_RECORDING_RATE - 1:
                vlosslist.append((valid_loss-last_loss)/LOSS_RECORDING_RATE)
                last_loss = valid_loss
                pbar.set_description(desc='net name: %s; loss: %.3f; validation loss: %.3f'
                                          % (k, running_loss/train_cnt, valid_loss/(j+1)))
                pbar.update()
    #print('Training Complete for %s; loss: %.3f; validation loss: %.3f' % (k, running_loss/train_cnt, valid_loss/validationcnt))

    loss_dictionary [k] = {'trainlosses':tlosslist,
                    'validlosses': vlosslist,
                    'trainlossavg': running_loss / train_cnt,
                    'validlossavg': valid_loss / validation_cnt}
    net.cpu()
    #torch.cuda.empty_cache()
    running_loss = 0.0
    last_loss = 0.0
print('Finished Training')
export_dict = network_dictionary.get_export_dict()
torch.save(export_dict, NET_PATH)
