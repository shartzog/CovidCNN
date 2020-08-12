'''
loss analysis and other model evaluation functions for NetDictionary objects
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import PIL
import seaborn as sns
from tqdm import tqdm
from network_dictionary_builder import NetDictionary

plt.ion()

class NetDictionaryAnalyzer():
    '''
    combines the loss dictionary objects for each net in a net dictionary into
    two consolidated dataframes to enable quick loss trend comparisons and summary
    reports 
    '''
    def __init__(self, net_dictionary, **kwargs):
        '''
        build summary and loss trending dataframes from net_dictionary data
        args:
            net_dictionary: a previously created and trained net_dictionary object
        '''
        assert isinstance(net_dictionary, NetDictionary), 'Invalid type: expected NetDictionary'
        assert net_dictionary._trained, 'Invalid NetDictionary: networks have not been trained'
        self.net_dictionary = net_dictionary
        self.net_losses_df = pd.DataFrame(columns=['net_number',
                                                   'iteration',
                                                   'loss',
                                                   'loss_type',
                                                   'optimizer_type'])
        self._loss_loc = 0
        self.__initialize_losses_df()

        self.net_summary_df = pd.DataFrame(columns=['net_number',
                                                    'param_count',
                                                    'conv_layer_count',
                                                    'has_pool',
                                                    'lin_layer_count',
                                                    'max_average_test_loss',
                                                    'max_loss_optimizer',
                                                    'max_loss_optimizer_efficiency',
                                                    'min_average_test_loss',
                                                    'min_loss_optimizer',
                                                    'min_loss_optimizer_efficiency'])
        self._sum_loc = 0
        self.__initialize_summary_df()

    def __initialize_losses_df(self):
        '''
        consolidate loss data for all networks into a single df for comparison trending.
        Populates net_losses_df which is also used to calucluate summary metrics.
        '''
        for k, net_d in self.net_dictionary.items():
            for i, loss in enumerate(net_d['loss_dictionary']['train_losses']):
                self.net_losses_df.loc[self._loss_loc] = pd.Series({'net_number':net_d['net_number'],
                                                         'iteration':loss[0],
                                                         'loss':loss[1],
                                                         'loss_type':'train',
                                                         'optimizer_type':net_d['optimizer_type']
                                                        })
                self._loss_loc += 1
            for i, loss in enumerate(net_d['loss_dictionary']['validation_losses']):
                self.net_losses_df.loc[self._loss_loc] = pd.Series({'net_number':net_d['net_number'],
                                                         'iteration':loss[0],
                                                         'loss':loss[1],
                                                         'loss_type':'valid',
                                                         'optimizer_type':net_d['optimizer_type']
                                                        })
                self._loss_loc += 1
        self.net_losses_df.net_number = self.net_losses_df.net_number.astype(int)


    def __initialize_summary_df(self):
        '''
        summarize the performance of each network for each optimizer type and calculate
        'network efficiencies' = (1/(trainable_param_count*average_test_loss))*10000.0. 
        Populates net_summary_df.
        '''
        last_net_num = 0
        param_count = 0
        conv_layers = 0
        has_pool = 0
        lin_layers = 0
        max_loss_optimizer = ''
        min_loss_optimizer = ''
        max_average_test_loss = 0.0
        min_average_test_loss = 1.0
        optimizer_type = ''
        net_d = dict()
        for k, net_d in self.net_dictionary.items():
            net_number = net_d['net_number']
            optimizer_type = net_d['optimizer_type']
            if last_net_num != net_number:
                self.__append_summary_row(net_d, series_dict)
                max_loss_optimizer = ''
                min_loss_optimizer = ''
                max_average_test_loss = 0.0
                min_average_test_loss = 1.0
            average_test_loss =  self.net_losses_df[self.net_losses_df.net_number == net_number
                                                   ][self.net_losses_df.loss_type == 'valid'
                                                    ][self.net_losses_df.optimizer_type == optimizer_type
                                                     ].loss.mean()
            if average_test_loss < min_average_test_loss:
                min_average_test_loss = average_test_loss
                min_loss_optimizer = optimizer_type
            if average_test_loss > max_average_test_loss:
                max_average_test_loss = average_test_loss
                max_loss_optimizer = optimizer_type
            last_net_num = net_number
            series_dict = {'net_number':net_number,
                           'max_average_test_loss':max_average_test_loss,
                           'max_loss_optimizer':max_loss_optimizer,
                           'min_average_test_loss':min_average_test_loss,
                           'min_loss_optimizer':min_loss_optimizer,
                          }
        self.__append_summary_row(net_d, series_dict)

    def __append_summary_row(self, net_dict_entry, series_dict):
        '''
        add row to summary dataframe
        '''
        param_count = sum(p.numel() for p in net_dict_entry['net'].parameters() if p.requires_grad)
        conv_layer_count = 0
        has_pool = 0
        lin_layer_count = 0
        max_valid_loss = series_dict['max_average_test_loss']
        min_valid_loss = series_dict['min_average_test_loss']
        for lyr in net_dict_entry['net'].lyrs:
            if str(lyr).startswith('Conv2d'):
                conv_layer_count += 1 
            if str(lyr).startswith('MaxPool'):
                has_pool = 1 
            if str(lyr).startswith('Linear'):
                lin_layer_count += 1                       
        series_dict.update({'max_loss_optimizer_efficiency':(1/(param_count*max_valid_loss))*10000.0,
                            'min_loss_optimizer_efficiency':(1/(param_count*min_valid_loss))*10000.0,
                            'param_count':param_count,
                            'param_count':param_count,
                            'conv_layer_count':conv_layer_count,
                            'has_pool':has_pool,
                            'lin_layer_count':lin_layer_count})
        self.net_summary_df.loc[self._sum_loc] = pd.Series(series_dict)
        self._sum_loc += 1
    
    def plot_losses(self, net_numbers=None):
        '''
        change to kwargs and allow user to specify hue and style
        '''
        if net_numbers == None:
            net_numbers = list({net_d_entry['net_number'] for k, net_d_entry in self.net_dictionary.items()})
        if len(net_numbers) > 9:
            net_numbers[:10]
        assert isinstance(net_numbers, list), 'net_numbers must be a list of integers'
        fig, axs = plt.subplots(len(net_numbers), 1, figsize=(20, 5*len(net_numbers)))
        for i, ax in enumerate(axs):
            sns.lineplot(ax=ax,
                         x='iteration',
                         y='loss',
                         hue='optimizer_type',
                         style='loss_type',
                         data=self.net_losses_df[self.net_losses_df.net_number==net_numbers[i]])
            ax.set_title('Conv Layers: ' + str(self.net_summary_df[self.net_summary_df.net_number==net_numbers[i]].conv_layer_count))


