"""
Utility functions for candidate_cnn_builder.py
"""
from __future__ import division, print_function

import math
import os.path
from random import random
from itertools import permutations
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import pyodbc
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(0)

#container for custom errors
class CustomError(Exception):
    """
    Dummy container for raising errors.
    """

class ImageTensorCreator:
    """
    utility class for translating binary column data into "image" tensor
    """
    def __init__(self, image_depth, data_columns):
        """
        set class variables for use in create_image_tensor
        args:
            image_depth:    image tensor depth.  must be between 1 (black and white) 
                            and 4 (RGBA) for generation of actual images
            data_columns:   tuple or list of names for binary data columns
        """
        self.image_depth = image_depth
        self.columns = data_columns
        self.col_count = len(data_columns)
        feature_count = 1.0
        #using image_depth - 1 to account for 'age' of patient in final dimension
        for i in range(self.image_depth - 1):
            #model will perform image_depth - 1 permutations for each column
            feature_count *= (self.col_count - i)
        self.image_height_width = math.ceil(feature_count**0.5)
        self.norm_tuple = list()
        for i in range(self.image_depth):
            self.norm_tuple.extend([0.5])
        self.norm_tuple = tuple(self.norm_tuple)
        print('total features: %i; image size: %i x %i x %i'
            % (feature_count, self.image_height_width, self.image_height_width, self.image_depth))

    def create_image_tensor(self, row):
        """
        Converts binary column data and patient age into an image tensor.
        INPUT:  row:    Expected format is [bindata,age] where bindata is a string of 1s and 0s
                        supplied by the DB to indicate patient conditions
        OUTPUT: Normalized image tensor
        """
        if row:
            icol = 0
            irow = 0
            if len(list(row[0])) != self.col_count:
                raise CustomError('Invalid Input Size: Expected %i, Received %i'
                                % (self.col_count, len(list(row[0]))))
            if len(row) < 2:
                raise CustomError('Invalid Data: Expected row format is [bindata,age]')
            image_data = np.zeros((self.image_height_width, self.image_height_width, self.image_depth),
                                dtype=np.float32)
            for pixel in permutations(list(row[0]), self.image_depth - 1):
                this_pixel = list()
                for channel in pixel:
                    this_pixel.extend([float(channel)])
                this_pixel.extend([float(row[1])/100.])
                image_data[icol, irow, :] = this_pixel
                irow += 1
                if irow > self.image_height_width - 1:
                    icol += 1
                    irow = 0
            return TF.normalize(TF.to_tensor(image_data), self.norm_tuple, self.norm_tuple)
        else:
            raise CustomError('Invalid Data: No row data supplied.')

    def create_fake_data(self, age, conditions):
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
            for col in self.columns:
                bindata += ('1' if col in conditions else '0')
            return self.create_image_tensor([bindata, age]).unsqueeze(0)
        total_conditions = 0
        if isinstance(conditions, int) and conditions < self.col_count:
            total_conditions = conditions
        else:
            total_conditions = math.floor(self.col_count*random())
        condition_count = 0
        for i,col in enumerate(self.columns):
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
        return self.create_image_tensor([bindata, age]).unsqueeze(0)

    def interpret_image_tensor(self, image_tensor):
        '''
        translate tensor to text description of patient
        args:
            image_tensor
        '''
        pts_data = list()
        for (bindata, age) in get_bindata_from_image_tensor(image_tensor):
            pt_data = str(int(age)) + ' yo, '
            for i, val in enumerate(list(bindata)):
                if i == 0:
                    if val == '1':
                        pt_data += 'Male, '
                    else:
                        pt_data += 'Female, '
                elif val == '1':
                    pt_data += self.columns[i] + ', '
            pt_data = pt_data[:-2]
            pts_data.append(pt_data)
        return pts_data

    def get_bindata_from_image_tensor(self, image_tensor):
        '''
        extract age and patient condition binary digit string from image tensor
        '''
        pts_binary_data = list()
        for case in image_tensor:
            case = case / 2 + 0.5
            case = case.numpy()
            case = np.transpose(case, (1, 2, 0))
            bindata = ''
            for i in range(self.image_depth - 1):
                bindata += str(int(case[0, 0, i]))
            for i in range(1, len(self.columns) - self.image_depth + 2):
                bindata += str(int(case[0, i, self.image_depth - 2]))
            age = float(round(float(case[0, 0, self.image_depth - 1])*100.))
            pts_binary_data.append((bindata, age))
        return pts_binary_data


class CovidCnnDataset(Dataset):
    '''
    download data from mexican covid database, reformat into image tensors
    and split into training and validation sets
    '''
    def __init__(self, import_export_filename, image_tensor_creator, **kwargs):
        '''
        download data from mexican covid database, reformat into image tensors
        and split into training and validation sets
        args:
            import_export_filename: file from which data is loaded if it exists and where data will
                                    be saved for future recall after the dataset is built
            image_tensor_creator:   instance of class ImageTensorCreator
        kwargs:
            pyodbc_conn_string:     ODBC connection string for database
            query:                  stored procedure or query used to create the cursor for the
                                    remote data
            force_rebuild:          force the dataset to rebuild even if import_export_filename
            mini_batch_size:        number of images to process before backpropagation
                                    and loss calculation
            truncate_data:          optionally trim dataset to a small size for testing/debugging
                                    on CPU.  default is True for CPU, False for cuda.
            truncate_train_size:    size of training dataset if truncation is enabled. 100
            approx_dataset_size:    approximate size of full dataset (train + validation)
            validation_ratio:       ratio of dataset to be used for validation, default 0.4
        '''
        super(CovidCnnDataset).__init__()
        self.import_export_filename = import_export_filename
        self.itc = image_tensor_creator
        self.conn_string = kwargs.get('pyodbc_conn_string', 'connection string not supplied')
        self.query = kwargs.get('query', 'query not supplied')
        self.force_rebuild = kwargs.get('force_rebuild', False)
        self.truncate_data = kwargs.get('truncate_data', False if DEVICE == 'cuda' else True)
        self.truncate_train_size = kwargs.get('truncate_train_size', 50)
        self.train_data = list()
        self.validation_images = list()
        self.validation_labels = list()
        if self.force_rebuild or not(os.path.exists(self.import_export_filename)):
            self.mini_batch_size = kwargs.get('mini_batch_size', 4)
            self.dataset_size = kwargs.get('approx_dataset_size',
                                           int(50000/self.mini_batch_size))
            self.validation_ratio = kwargs.get('validation_ratio', 0.4)
            self.validation_count = int(self.validation_ratio*self.dataset_size)
            self.train_count = self.dataset_size - self.validation_count
            rowgen = self.next_row()
            if self.truncate_data:
                self.train_count = self.truncate_train_size
            pbar = tqdm(enumerate(rowgen), total=self.train_count)
            for i, data in pbar:
                self.train_data.append(data)
            self.save_to_disk()
        else:
            self.reload_from_disk()
        self.dataset_size = len(self.validation_images) + len(self.train_data)
        self.train_count = len(self.train_data)
        self.validation_count = len(self.validation_images)
        self.validation_ratio = round(float(len(self.validation_images))
                                      /float(self.dataset_size),3)

    def __len__(self):
        '''
        required overload
        '''
        return self.train_count

    def __getitem__(self, idx):
        '''
        required overload.  returns 'idx'th training sample in the form:
        a tensor of [self.mini_batch_size] image tensors,
        a tensor of [self.mini_batch_size] label tensors
        '''
        return train_data[idx]

    def save_to_disk(self):
        '''
        save dataset to disk
        '''
        torch.save({'train_data' : self.train_data,
                    'validation_images' : self.validation_images,
                    'validation_labels': self.validation_labels,
                    'mini_batch_size': self.mini_batch_size,
                    }, self.import_export_filename)

    def reload_from_disk(self):
        '''
        save any manual edits performed in external subroutines to disk
        '''
        datasets = torch.load(self.import_export_filename)
        self.train_data = datasets['train_data']
        self.validation_images = datasets['validation_images']
        self.validation_labels = datasets['validation_labels']
        self.mini_batch_size = datasets['mini_batch_size']

    def next_row(self):
        """
        Generator function for retrieving patient data.
        OUTPUT: tuple of form (tensor of [BATCH_SIZE] images, tensor of [BATCH_SIZE] labels)
        """
        conn = pyodbc.connect(self.conn_string)
        crsr = conn.cursor()
        crsr.execute(self.query)
        rowcnt = 0
        imgs = list()
        labels = list()
        row = [0, 1, 2]
        idx = 0
        while row and (not self.truncate_data or len(self.train_data) < self.truncate_train_size):
            while rowcnt < self.mini_batch_size:
                try:
                    row = crsr.fetchone()
                except:
                    conn = pyodbc.connect(self.conn_string)
                    crsr = conn.cursor()
                    crsr.execute(self.query)
                    crsr.skip(idx*self.mini_batch_size + rowcnt)
                    row = crsr.fetchone()
                rowcnt += 1
                if row:
                    imgtensor = self.itc.create_image_tensor(row)
                    labels.append(np.array(list(row[2]), dtype=np.float32))
                    imgs.append(imgtensor)
                else:
                    break
            if row:
                if random() < self.validation_ratio:
                    self.validation_images.append(torch.stack(imgs))
                    self.validation_labels.append(torch.tensor(labels))
                else:
                    yield torch.stack(imgs), torch.tensor(labels)
                rowcnt = 0
                idx += 1
                imgs = list()
                labels = list()
            else:
                break

        