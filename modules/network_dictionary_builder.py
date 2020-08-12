"""
Contains Net and NetDictionary class for creating a random collection of CNN structures
or loading a previously created collection.
"""
from __future__ import division, print_function

from random import random
import os.path

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
from numpy.random import randint as r_i
from tqdm import tqdm

DEBUG = False   #prints tensor size after each network layer during network creation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(0)
else:
    print('**** CUDA not available - continuing with CPU ****')

#global classes
class Net(nn.Module):
    """
    Build pytorch module using eval() on incoming model tensor and lists of eval strings
    for layers and params.
    """
    def __init__(self, modelinputtensor, layerlist, layerparams, **kwargs):
        """
        args:
        modelinputtensor:   example model input tensor (including an arbitrary batch dimension)
        layerlist:  list of pytorch nn fucntions as their 'F' namespace equivalents
                    Example: 'nn.MaxPool2d' should be supplied as 'F.max_pool2d'
        layerparams:    list of _independent_ params in their nn form and passed as a tuple.
        kwargs:
        activations:    list of activation functions for forward layers.  the length
                        of the list must match the length of layerlist exactly even
                        though the activation function supplied for any pooling
                        layers will be ignored and the final value supplied will
                        always be replaced by Sigmoid.
        Example:
            The first conv2d layer will have 3 params in a tuple
                of form (in_channels, out_channels, kernel_size).
            Subsequent conv2d layers will have _2_ params in a tuple
                of form (out_channels, kernel_size) since the in_channels
                are determined by the previous layer.
            Pooling layers will always have params of the form (x, y)
                corresponding to the pooling window size.
            Linear layers will always have a single param corresponding to
                the number of out features for the layer since input
                features are determined by the preceding layer)
        """
        super(Net, self).__init__()
        self.activations = kwargs.get('activations', ['F.relu' for layer in layerlist])
        self.lyrs, self.fwdlyrs = self.get_layers(modelinputtensor, layerlist, layerparams, self.activations, DEBUG)

    def forward(self, x):
        """
        """
        for f in self.fwdlyrs:
            x = eval(f)
        return torch.sigmoid(x)

    def get_layers(self, testtensor, funcs, params, activations, debug):
        """
        Build network layers from supplied test tensor, funcs, and param eval strings.
        """
        initlayers = nn.ModuleList()
        fwdlayers = list()
        if debug == 1:
            print(testtensor.size())
        lastsize = testtensor.size()
        lastsize = None
        lyr = 0
        with torch.no_grad():
            for fn, pa in zip(funcs, params):
                if lastsize is not None:
                    if fn.__name__ == 'conv2d':
                        pa = (lastsize[1], pa[0], pa[1])
                    elif fn.__name__ == 'linear':
                        if not testtensor.ndim == 2:
                            testtensor = testtensor.view(-1, self.num_flat_features(testtensor))
                            fwdlayers.append("x.view(-1,self.num_flat_features(x))")
                            lastsize = testtensor.size()
                        pa = (lastsize[1], pa)
                if fn.__name__ == 'conv2d':
                    paeval = ",".join(tuple(map(str, (pa[1], pa[0], pa[2], pa[2]))))
                    paeval = "torch.tensor(np.random.rand(" + paeval + "), dtype=torch.float32)"
                elif fn.__name__ == 'max_pool2d':
                    paeval = ",".join(tuple(map(str, pa)))
                elif fn.__name__ == 'linear':
                    paeval = ",".join(tuple(map(str, (pa[1], pa[0]))))
                    paeval = "torch.tensor(np.random.rand(" + paeval + "),dtype=torch.float32)"
                if not fn.__name__ == 'linear' or pa[0] > pa[1]:
                    testtensor = fn(testtensor, eval(paeval))
                    lastsize = testtensor.size()
                    initlayers.append(eval(self.__get_init_equivalent(fn.__name__, pa)))
                    fwdlayers.append(self.__get_fwd_equivalent(fn.__name__, lyr))
                    lyr += 1
                    if debug == 1:
                        print(testtensor.size())
                elif debug == 1:
                    print('NetDictionary: Eliminating linear layer - out features > previous layer')
        fwdlayers[-1] = 'self.lyrs[' + str(lyr - 1) + '](x)'
        return initlayers, fwdlayers

    def num_flat_features(self, x):
        """
        Calculate number of flat features in a given net layer.
        Useful for transitioning between conv and linear layers.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __get_init_equivalent(self, funcname, initparams):
        """
        Construct eval string from supplied funtions and parameters for the
        style required in the torch.nn.Module __init__.
        """
        return 'nn.' + ''.join([val.capitalize()
                                for val in funcname.split('_')
                               ]) + '(' + ",".join(tuple(map(str, initparams))) + ')'

    def __get_fwd_equivalent(self, funcname, lyrnum):
        """
        Construct eval string from supplied funtions and parameters for the
        style required in the torch.nn.Module __init__.
        """
        if not funcname == 'max_pool2d':
            return self.activations[lyrnum] + '(self.lyrs[' + str(lyrnum) + '](x))'
        else:
            return 'self.lyrs[' + str(lyrnum) + '](x)'



class NetDictionary(dict):
    """
    Holds a dictionary of Net with functions to build a model tensor and
    random layer and param lists
    """
    def __init__(self, network_count, test_tensor, total_labels, import_export_filename, **kwargs):
        """
        Initialize a dictionary of randomly structured CNNs to test various network configurations.
        args:   network_count:  number of networks to generate
                test_tensor:    a tensor that can be used to construct network layers
                total_labels:   the number of labels being predicted for the networks
                import_export_filename: if file exists on initialization, the information in the
                                        file will be used to reconstruct a prior network.
        kwargs: optimizers: list of tuples of form (eval strings for optimizer creation, label)
                default=[("optim.SGD(d['net'].parameters(), lr=0.0001, momentum=0.9)", "SGD"),
                         ("optim.Adam(d['net'].parameters(), lr=0.0001)", "Adam"),
                         ("optim.Adam(d['net'].parameters(), lr=0.00001)", "Adam1")]
                force_rebuild:  override import from file and recreate network even if
                                import_export_filename already exists, false
                force_training: imported network training is bypassed if set to false
                                default=True 
                conv_layer_activation:  activation function used by conv layers
                                default=F.relu
                pooling_probability:    approximate fraction of random networks that are assigned
                                        a pooling layer, default = 0.5
                first_conv_layer_depth, 4
                max_conv_layers, 5
                min_conv_layers, 1
                max_kernel_size, 7
                min_kernel_size, 3
                max_out_channels, 12
                min_out_channels, 4
                linear_layer_activation, F.relu
                init_linear_out_features, 1000
                linear_feature_deadband, 20
                max_layer_divisor, 20
                min_layer_divisor, 4
        """
        super(NetDictionary, self).__init__()
        self.net_count = network_count
        self.label_count = total_labels
        self.import_export_filename = import_export_filename
        self.__test_tensor = test_tensor
        self._trained = False
        self.force_rebuild = kwargs.get('force_rebuild', False)
        self.force_training = kwargs.get('force_training', True)
        self.pooling_probability = kwargs.get('pooling_probability', 0.5)
        self.init_from_file = os.path.exists(import_export_filename) and not self.force_rebuild
        if self.init_from_file and not self.force_rebuild:
            self.__import_networks()
        else:
            self.__build_networks(**kwargs)

    def __import_networks(self):
        """
        Read layer info and net state dicts from disk.
        """
        net_info = torch.load(self.import_export_filename)
        self.__options = net_info['options'].copy()
        self.optimizers = self.__options['optimizers']
        for n_key, n_dict in net_info['state_dicts'].items():
            d = dict()
            d['net_number'] = net_info['net_numbers'][n_key]
            d['func_list'] = net_info['func_lists'][n_key]
            d['params'] = net_info['params'][n_key]
            d['activations'] = net_info['activations'][n_key]
            funcs = [eval(f) for f in d['func_list']]
            d['net'] = Net(self.__test_tensor, funcs, d['params'],activations=d['activations'])
            d['net'].load_state_dict(n_dict)
            d['optimizer_type'] = net_info['optimizer_types'][n_key]
            d['criterion'] = nn.BCELoss()
            d['optimizer'] = eval([optim[0] for optim in self.optimizers if optim[1] == d['optimizer_type']][0])
            d['loss_dictionary'] = net_info['loss_dictionaries'][n_key]
            self.__setitem__(n_key, d)
        self._trained = True

    def __build_networks(self, **kwargs):
        """
        build a new set of randomized networks
        """
        self.__options = {
            'optimizers': kwargs.get('optimizers',
                                     [("optim.SGD(d['net'].parameters(), lr=0.0001, momentum=0.9)",
                                       "SGD"),
                                      ("optim.Adam(d['net'].parameters(), lr=0.0001)", "Adam"),
                                      ("optim.Adam(d['net'].parameters(), lr=0.00001)", "Adam1"),
                                     ]),
            'convolution_layer_options': {
                'activation' : kwargs.get('conv_layer_activation', 'F.relu'),
                'first_layer_depth' : kwargs.get('first_conv_layer_depth', 4),
                'max_layers' : kwargs.get('max_conv_layers', 5),
                'min_layers' : kwargs.get('min_conv_layers', 1),
                'max_kernel_size' : kwargs.get('max_kernel_size', 7),
                'min_kernel_size' : kwargs.get('min_kernel_size', 3),
                'max_out_channels' : kwargs.get('max_out_channels', 12),
                'min_out_channels' : kwargs.get('min_out_channels', 4),
            },
            'linear_layer_options': {
                'activation' : kwargs.get('linear_layer_activation', 'F.relu'),
                'init_out_features' : kwargs.get('init_linear_out_features', 1000),
                'feature_deadband' : kwargs.get('linear_feature_deadband', 20),
                'max_layer_divisor' : kwargs.get('max_layer_divisor', 20),
                'min_layer_divisor' : kwargs.get('min_layer_divisor', 4),
            },
        }
        self.optimizers = self.__options['optimizers']
        for i in tqdm(range(self.net_count)):
            cfs, cps = self.__get_convolution_layers(self.__options['convolution_layer_options'])
            lfs, lps = self.__get_linear_layers(self.__options['linear_layer_options'])
            funcs = cfs
            params = cps
            activations = [self.__options['convolution_layer_options']['activation'] for f in cfs]
            if (random() < self.pooling_probability):
                funcs.extend([F.max_pool2d])
                pool_size = np.random.randint(2,4)
                activations.extend(['F.relu'])
                params.extend([(pool_size, pool_size)])
            funcs.extend(lfs)
            activations.extend([self.__options['linear_layer_options']['activation'] for f in lfs])
            func_list = ['F.' + f.__name__ for f in funcs]
            params.extend(lps)
            for opt in self.optimizers:
                d = dict()
                d['net'] = Net(self.__test_tensor, funcs, params, activations=activations)
                d['net_number'] = i
                d['func_list'] = func_list
                d['params'] = params
                d['activations'] = activations
                d['optimizer_type'] = opt[1]
                d['criterion'] = nn.BCELoss()
                d['optimizer'] = eval(opt[0])
                self.__setitem__(str(i) + '-' + opt[1], d)

    def __get_convolution_layers(self, c):
        """
        Dynamically create a list of convolution layers.  Parameters are used to manage the size,
        complexity, and structure of each layer.  NEEDS IMPROVEMENT.
        """
        fncs, parms = list(),list()
        fncs.append(F.conv2d)
        parms.append((c['first_layer_depth'],
                      r_i(c['min_out_channels'], c['max_out_channels'] + 1),
                      r_i(c['min_kernel_size'], c['max_kernel_size']+1)))
        for i in range(r_i(c['min_layers']-1,c['max_layers'])):
            fncs.append(F.conv2d)
            parms.append((r_i(c['min_out_channels'],c['max_out_channels'] + 1),
                          r_i(c['min_kernel_size'], c['max_kernel_size']+1)))
        return fncs, parms

    def __get_linear_layers(self, d):
        """
        Dynamically create a list of linear layers.
        Parameters are used to manage the size of each layer.
        NEEDS IMPROVEMENT.
        """
        fncs, parms = list(), list()
        fncs.append(F.linear)
        parms.append(d['init_out_features'])
        nextoutfeatures = int(d['init_out_features']/r_i(d['min_layer_divisor'],
                                                         d['max_layer_divisor'] + 1)) 
        while nextoutfeatures > self.label_count + d['feature_deadband']:
            fncs.append(F.linear)
            parms.append(nextoutfeatures)
            nextoutfeatures = int(nextoutfeatures/r_i(d['min_layer_divisor'],
                                                      d['max_layer_divisor'] + 1))
        fncs.append(F.linear)
        parms.append(self.label_count)
        return fncs,parms

    def train_validate_networks(self, train_data, validation_images, validation_labels, loss_recording_rate):
        for k, d in self.items():
            net = d['net']
            net.to(DEVICE)
            net.train()
            criterion = d['criterion']
            optimizer = d['optimizer']
            train_losses = []
            validation_losses = []
            last_loss = 0.0
            running_loss = 0.0
            if self.force_training or not self.init_from_file:
                pbar = tqdm(enumerate(train_data), total=len(train_data))
                for i, data in pbar:
                    #get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    #zero the parameter gradients
                    optimizer.zero_grad()
                    #forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if i % loss_recording_rate == loss_recording_rate - 1:
                        train_losses.append((i + 1, (running_loss - last_loss)/loss_recording_rate))
                        pbar.set_description(desc='net name: %s; loss: %.3f' % (k, running_loss/(i + 1)))
                        pbar.update()
                        last_loss = running_loss
            last_loss = 0.0
            valid_loss = 0.0
            net.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(zip(validation_images,validation_labels)),total=len(validation_labels))
                for j, (v_in, v_lab) in pbar:
                    v_in, v_lab = v_in.to(DEVICE), v_lab.to(DEVICE)
                    outputs = net(v_in)
                    loss = criterion(outputs, v_lab)
                    valid_loss += loss.item()
                    if j % loss_recording_rate == loss_recording_rate - 1:
                        validation_losses.append((j + 1, (valid_loss - last_loss)/loss_recording_rate))
                        last_loss = valid_loss
                        pbar.set_description(desc='net name: %s; loss: %.3f; validation loss: %.3f'
                                                % (k, running_loss/len(train_data), valid_loss/(j + 1)))
                        pbar.update()

            self[k]['loss_dictionary'] = {'train_losses':train_losses,
                                          'validation_losses': validation_losses,
                                         }
            net.cpu()
        self._trained = True


    def export_networks(self):
        """
        Write info required to reconstruct this NetDictionary to disk.
        """
        state_dicts = {key : d['net'].state_dict() for key, d in self.items()}
        net_numbers = {key : d['net_number'] for key, d in self.items()}
        func_lists = {key : d['func_list'] for key, d in self.items()}
        params = {key : d['params'] for key, d in self.items()}
        activations = {key : d['activations'] for key, d in self.items()}
        optimizer_types = {key : d['optimizer_type'] for key, d in self.items()}
        loss_dictionaries = {key : d['loss_dictionary'] for key, d in self.items()}
        torch.save({'state_dicts':state_dicts,
                    'net_numbers':net_numbers,
                    'func_lists':func_lists,
                    'params':params,
                    'activations':activations,
                    'optimizer_types':optimizer_types,
                    'options':self.__options,
                    'test_tensor':self.__test_tensor,
                    'loss_dictionaries':loss_dictionaries,
                   }, self.import_export_filename)