"""
Contains Net and NetDictionary class for creating a random collection of CNN structures
or loading a previously created collection.
"""
from random import random
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm

#global classes
class Net(nn.Module):
    """
    Build pytorch module using eval() on incoming model tensor and lists of eval strings
    for layers and params.
    """
    def __init__(self, modelinputtensor, layerlist, layerparams):
        """
        torch.nn must be imported as nn
        torch.nn.functional must be imported as F
        modelinputtensor = example model input tensor (including an arbitrary batch dimension)
        layerlist = list of pytorch nn fucntions as their 'F' namespace equivalents
                    Example: 'nn.MaxPool2d' should be supplied as 'F.max_pool2d'
        layerparams = list of _independent_ params in their nn form and passed as a tuple.
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
        self.lyrs, self.fwdlyrs = self.get_layers(modelinputtensor, layerlist, layerparams)

    def forward(self, x):
        """
        """
        for f in self.fwdlyrs:
            x = eval(f)
        return torch.sigmoid(x)

    def get_layers(self, testtensor, funcs, params, debug=0):
        """
        Build network layers from supplied test tensor and func/param eval strings.
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
            return 'F.relu(self.lyrs[' + str(lyrnum) + '](x))'
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
                first_conv_layer_depth
                max_conv_layers
                min_conv_layers
                max_kernel_size
                min_kernel_size
                max_out_channels
                min_out_channels
                init_linear_out_features
                linear_feature_deadband
                max_layer_divisor
                min_layer_divisor
        """
        super(NetDictionary, self).__init__()
        self.net_count = network_count
        self.label_count = total_labels
        self.import_export_filename = import_export_filename
        self.__test_tensor = test_tensor
        self.init_from_file = os.path.exists(import_export_filename)
        if self.init_from_file:
            self.__import_networks()
        else:
            self.__build_networks(**kwargs)

    def __import_networks(self):
        """
        Read layer info and net stat dicts from disk.
        """
        net_info = torch.load(self.import_export_filename)
        self.__options = net_info['options'].copy()
        self.optimizers = self.__options['optimizers']
        for n_key, n_dict in net_info['state_dicts']:
            d = dict()
            d['net_number'] = net_info['net_numbers'][n_key]
            d['funcs'] = net_info['funcs'][n_key]
            d['params'] = net_info['params'][n_key]
            d['net'] = Net(self.__test_tensor, d['funcs'], d['params'])
            d['net'].load_state_dict(n_dict)
            d['optimizer_type'] = net_info['optimizer_types'][n_key]
            d['criterion'] = nn.BCELoss()
            d['optimizer'] = eval([optim[0] for optim in self.optimizers if optim[1] == d['optimizer_type']][0])
            d['loss_dictionary'] = net_info['loss_dictionaries'][n_key]
            self.__setitem__(n_key, d)

    def __build_networks(self, **kwargs):
        self.__options = {
            'optimizers': kwargs.get('optimizers',
                                     [("optim.SGD(d['net'].parameters(), lr=0.0001, momentum=0.9)",
                                       "SGD"),
                                      ("optim.Adam(d['net'].parameters(), lr=0.0001)", "Adam"),
                                      ("optim.Adam(d['net'].parameters(), lr=0.00001)", "Adam1"),
                                     ]),
            'convolution_layer_options': {
                'first_layer_depth' : kwargs.get('first_conv_layer_depth', 4),
                'max_layers' : kwargs.get('max_conv_layers', 5),
                'min_layers' : kwargs.get('min_conv_layers', 1),
                'max_kernel_size' : kwargs.get('max_kernel_size', 7),
                'min_kernel_size' : kwargs.get('min_kernel_size', 3),
                'max_out_channels' : kwargs.get('max_out_channels', 12),
                'min_out_channels' : kwargs.get('min_out_channels', 4),
            },
            'linear_layer_options': {
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
            if random() > 0.3:
                funcs.extend([F.max_pool2d])
                poolsize = np.random.randint(2,4)
                params.extend([(poolsize, poolsize)])
            funcs.extend(lfs)
            params.extend(lps)
            for opt in self.optimizers:
                d = dict()
                d['net'] = Net(self.__test_tensor, funcs, params)
                d['net_number'] = i
                d['funcs'] = funcs
                d['params'] = params
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
        r_i = np.random.randint
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
        r_i = np.random.randint
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

    def export_networks(self):
        """
        Write info required to reconstruct this NetDictionary to disk.
        """
        state_dicts = {key : d['net'].state_dict() for key, d in self.items()}
        net_numbers = {key : d['net_number'] for key, d in self.items()}
        funcs = {key : d['funcs'] for key, d in self.items()}
        params = {key : d['params'] for key, d in self.items()}
        optimizer_types = {key : d['optimizer_type'] for key, d in self.items()}
        loss_dictionaries = {key : d['loss_dictionary'] for key, d in self.items()}
        torch.save({'state_dicts':state_dicts,
                    'net_numbers':net_numbers,
                    'funcs':funcs,
                    'params':params,
                    'optimizer_types':optimizer_types,
                    'options':self.__options,
                    'loss_dictionaries':loss_dictionaries,
                   }, self.import_export_filename)