# %% 
''' Now that I have the algo working ok with META in the file './META-loader/src/porting_contextual_to_seq...'
Now I want to see if I can easily switch between the two tasks.'''

#!%load_ext autoreload
#!%autoreload 2

import torch
from torch.utils.data import Dataset, DataLoader   
import torch.nn as nn

import os
import numpy as np
import matplotlib as mpl
# make text smaller than the default
# mpl.rcParams['font.size'] = 8
# make the axis lines thinner
mpl.rcParams['axes.linewidth'] = 0.8

import seaborn as sns
sns.set(font_scale=0.8) # could not get rid of sns changing font size all over the META code. So now just embracing it.
sns.set_style('white', {'axes.linewidth':0.5}) # to remove grid
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 0.5
# remove spines on right and top
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Image

# add META.src to the path
# import sys
# sys.path.append('./META/src/')
from META.src.utils_meta import to_pth, pickle_load_dict
# from scipy.cluster.hierarchy import dendrogram, linkage
from META.src.task._METAConstants import hier_struct, hlc2hlcid, mlc2mlcid, llc_names, \
    llc2llcid, llc2mlcid, llc2hlcid, llcid2llc, llcid2hlcid, llcid2mlcid
from META.src.task._METAVideos import *
from META.src.task._METAConstants import *

# sns.set(style='white', palette='colorblind', context='talk')
# cpal = sns.color_palette('colorblind')

from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from functions_and_utils import *

class Config:
    def __init__(self, experiment_name = 'contextual_switching_task'):
        self.experiment_name = experiment_name
        self.export_path = f'./{experiment_name}/exports/'
        if experiment_name == 'contextual_switching_task':
            self.input_size = 1 # gets updated below
            self.hidden_size = 32
            self.output_size = 1
            self.seq_len = 10 #300
            self.stride = 1 # self.seq_len
            # training
            self.passive_epochs = 1
            self.epochs = 2
            self.batch_size = 100

        elif experiment_name == 'META':
            self.input_size = 30 # gets updated below
            self.hidden_size = 200
            self.output_size = 30
            self.seq_len = 10
            self.stride = 1 # self.seq_len
            # training
            self.passive_epochs = 1
            self.epochs = 2
            self.batch_size = 100
            self.test_split = 0.2

        # self.save_model = False
        self.save_model = True
        self.load_saved_model = not self.save_model
        self.limited_testing_samples_no = 50

        self.WU_lr = 0.001
        self.no_of_frames_to_prompt_generate = 10
        self.predict_first_frame = True
        self.add_noise_to_input = False
        self.noise_std = 0.3

        # latent
        self.LU_lr = 0.01
        self.latent_type = '1d_latent'
        self.latent_dims = [10] 
        self.latent_chuncks = 2 # how many chuncks to divide the latent into.
        self.latent_activation = 'sigmoid' #'softmax'
        self.latent_activation_dim = 0 # not used
        self.latent_activation_temp = 1 # not used for sigmoid
        self.momentum = 0.9 # only for sgd
        self.l2_loss = 0 #0.001 # weight decay for Adam
        self.LU_optimizer = 'Adam' # 'SGD' 'Adam'
        self.input_size += np.prod(self.latent_dims) # update the input size to include the latent
        self.loss_reduction_LU = 'mean' # 'sum' 'mean' 'none'
        self.loss_reduction_WU = 'mean' # 'mean' 'none'
        self.allow_latent_updates = True # Truned off to simulate passive learning without latent updates.

        # latent config for latent I and II [I, II]
        self.latent_averaging_window = [3, 20] # how many frames to average the latent over
        self.latent_value_to_use = ['last', 'first'] # 'average' 'filtered' # updates the latent value witht the grad of the first or last element of the sequence.
        self.no_of_steps_in_latent_space = 100 
        self.no_of_steps_in_weight_space = 1
        # self.latent_aggregation_op = 'none' # 'last' 'first' 'average' 'none' # how to aggregate the gradients of the latent over the sequence
        self.latent_aggregation_op = 'average' # 'last' 'first' 'average' 'none' # how to aggregate the gradients of the latent over the sequence

        # Contexutal task
        self.no_of_blocks = 50
        self.block_size = 50
        self.latent_change_interval = 1
        self.high_level_latent_change_interval_in_blocks = 3
        self.default_std = 0.1

        # META params


        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class METAdataset(Dataset):
    def __init__(self, metavideos, config, no_of_videos_to_use=0, to_torch=True, split='training', get_holdout=False):

        self.metavideos = metavideos
        self.config = config
        if not get_holdout:
            self.X_list, self.llc_name_list = metavideos.get_data(to_torch=True)
        else:
            self.X_list, self.llc_name_list = metavideos.get_holdout_data(to_torch=True)
        if no_of_videos_to_use > 0:
            self.X_list = self.X_list[:no_of_videos_to_use]
            self.llc_name_list = self.llc_name_list[:no_of_videos_to_use]
        self.llc2llcid = metavideos.llc2llcid
        self.llc2hlcid = metavideos.llc2hlcid

        # Concatenate all videos into one tensor
        self.concatenated_X = torch.cat(self.X_list, dim=0)

        # Create tensors with llc and hlc for each frame
        self.llc_ids = []
        self.hlc_ids = []
        # Loop through the videos and create tensors with llc and hlc for each frame
        for i, X in enumerate(self.X_list):
            llc_id = self.llc2llcid[self.llc_name_list[i]]
            hlc_id = self.llc2hlcid[self.llc_name_list[i]]
            self.llc_ids.append(torch.full((X.shape[0],), llc_id, dtype=torch.long))
            self.hlc_ids.append(torch.full((X.shape[0],), hlc_id, dtype=torch.long))
        # Concatenate the tensors
        self.llc_ids = torch.cat(self.llc_ids, dim=0)
        self.hlc_ids = torch.cat(self.hlc_ids, dim=0)

        # move to config.device
        self.concatenated_X = self.concatenated_X.to(config.device)
        self.llc_ids = self.llc_ids.to(config.device)
        self.hlc_ids = self.hlc_ids.to(config.device)

        # Split the dataset into training and test sets
        if split == 'training':
            split_index = int(len(self.concatenated_X) * (1 - config.test_split))
            self.concatenated_X = self.concatenated_X[:split_index]
            self.llc_ids = self.llc_ids[:split_index]
            self.hlc_ids = self.hlc_ids[:split_index]
        elif split == 'test':
            split_index = int(len(self.concatenated_X) * (1 - config.test_split))
            self.concatenated_X = self.concatenated_X[split_index:]
            self.llc_ids = self.llc_ids[split_index:]
            self.hlc_ids = self.hlc_ids[split_index:]

    def __len__(self):
        return (self.concatenated_X.shape[0] - self.config.seq_len) // self.config.stride + 1

    # getitem now should take a fid and return a sequence of frames
    def __getitem__(self, fid):
        ''' fid is the frame id to start from and return a sequence of frames of length seq_len'''
        start = fid * self.config.stride
        end = start + self.config.seq_len
        X = self.concatenated_X[start:end]
        llc_id = self.llc_ids[start:end]
        hlc_id = self.hlc_ids[start:end]
        return X, llc_id, hlc_id


class RNN_with_latent(nn.Module):
    def __init__(self, config):
        super(RNN_with_latent, self).__init__()
        self.config = config
        
        self.input_layer = nn.Linear(config.input_size, config.hidden_size)
        self.lstm_cell = nn.LSTMCell(config.hidden_size, config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)


        self.init_hidden()
        self.init_latent()
        self.WU_optimizer = self.get_WU_optimizer()
        self.LU_optimizer = self.get_LU_optimizer()

    def init_hidden(self, batch_size=None): 
        if batch_size is None:
            batch_size = self.config.batch_size
        self.hidden_state = torch.zeros(batch_size, self.config.hidden_size, device=self.config.device)
        self.cell_state = torch.zeros(batch_size, self.config.hidden_size, device=self.config.device)

    def init_latent(self, batch_size=None, seq_len=None):
        ''' initializes the latent variable'''
        if batch_size is None:
            batch_size = self.config.batch_size
        if seq_len is None:
            seq_len = self.config.seq_len

        if self.config.latent_type == 'one_latent': # one latent for the whole sequence
            latent_size = np.prod(self.config.latent_dims)
            self.register_parameter('latent', nn.Parameter(torch.ones(batch_size, 1, self.config.latent_dims)/latent_size, requires_grad=True, ))
        elif self.config.latent_type == '1d_latent':
            self.register_parameter('latent', nn.Parameter(torch.ones(batch_size, seq_len, self.config.latent_dims[0])/self.config.latent_dims[0], requires_grad=True, ))
        else:
            raise ValueError('Invalid latent_type')
        self.latent.data = self.latent.data.to(self.config.device)

        # reattach the optimizer to the new latent
        self.LU_optimizer = self.get_LU_optimizer()
    
    def reset_latent(self, batch_size=None, seq_len=None):
        ''' zeros the latent'''
        if batch_size is None:
            batch_size = self.config.batch_size
        if seq_len is None:
            seq_len = self.config.seq_len

        if batch_size != self.latent.shape[0] or seq_len != self.latent.shape[1]:
            self.init_latent(batch_size, seq_len)

        self.latent.data = torch.zeros_like(self.latent.data)

    def get_WU_optimizer(self):
        weights = [p for n,p in self.named_parameters() if n !='latent']
        WU_optimizer = torch.optim.Adam(weights, lr=self.config.WU_lr)
        return WU_optimizer

    def get_LU_optimizer(self):
        if self.config.LU_optimizer == 'Adam':
            LU_optimizer = torch.optim.Adam([self.latent], lr=self.config.LU_lr, weight_decay= self.config.l2_loss if self.config.l2_loss else 0)
        elif self.config.LU_optimizer == 'SGD':
            LU_optimizer = torch.optim.SGD([self.latent], lr=self.config.LU_lr, momentum=self.config.momentum)
        return LU_optimizer
    
    def latent_activation_function(self, x):
        if self.config.latent_activation == 'softmax':
            return torch.softmax(x/self.config.latent_activation_temp, dim=self.config.latent_activation_dim)
        elif self.config.latent_activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.config.latent_activation == 'none':
            return x
        else:
            raise ValueError('Invalid latent_activation')
            
    def forward(self, input):
        batch_size = input.size(0)
        self.init_hidden(batch_size)
        hidden_state = self.hidden_state; cell_state = self.cell_state
        outputs = []
        input = self.input_layer(input)


        for i in range(self.config.seq_len):
            hidden_state, cell_state = self.lstm_cell(input[:, i, ...], (hidden_state, cell_state))
            output = self.output_layer(hidden_state)
            outputs.append(output)
            
        return outputs, (hidden_state, cell_state)
    
    def combine_input_with_latent(self, input, what_latent = 'self', taskID=None):
        ''' Cats current input with current self.latent and returns updated input
            
            Types of latent to combine:
            - uniform
            - task ID, in wich case ground truth task representation or ID should be provided.
            - Current self.latent
                self.latent can be of different sizes. 
                First is batch = 1 or batch_size  
                seq = 1 or seq_len 
            - 
            what_latent: 'self' or 'uniform' 'taskID'
            '''
        input_shape = input.shape
        _latent = self.latent

        if what_latent == 'self': 
            if self.config.latent_type == 'one_latent':
                raise NotImplementedError('one_latent was instead implemented as averaging grad op over the sequence. Use latent_aggregation_op = average instead.')
                if len(_latent.shape) == 1 and len(input_shape) == 3:
                    _latent = _latent.unsqueeze(0).unsqueeze(1).repeat(input_shape[0], 1, 1)
                # expand the latent to match the input shape[1]
                    # IMPORTANT using expand instgead of repeat to collect all the gradients into one value
                if input_shape[1] > _latent.shape[1]: # input has seq_len > 1
                    _latent = _latent.expand(input_shape[0], input_shape[1], *_latent.shape[2:])
            else:
                if len(_latent.shape) == 1 and len(input_shape) == 3:
                    _latent = _latent.unsqueeze(0).unsqueeze(1).repeat(input_shape[0], input_shape[1], 1)
                elif input_shape[0] > _latent.shape[0]: # input batch size increased, repeat the last batch latent and append to match input batch size
                    print('shape of input {} and latent {}'.format(input.shape, _latent.shape))
                    _latent = torch.cat((_latent, _latent[-1, ...].repeat(input_shape[0] - _latent.shape[0], 1, 1)), dim=0)
                elif input_shape[0] < _latent.shape[0]: # input batch size decreased
                    print('shape of input {} and latent {}'.format(input.shape, _latent.shape))
                    _latent = _latent[-input_shape[0]:, ...]
                elif input_shape[1] > _latent.shape[1]: # input has seq_len > 1
                    _latent = _latent.repeat(1, input_shape[1], *_latent.shape[2:])
            
        elif what_latent == 'uniform':
            _latent= torch.ones((*input_shape[:-1], _latent.shape[-1])) * 1/np.prod(_latent.shape[-1], device=self.config.device)
        elif what_latent == 'zeros':
            _latent = torch.zeros((*input_shape[:-1], _latent.shape[-1]), device=self.config.device)
        elif what_latent == 'init':
            _latent = torch.ones_like(_latent) * 1/np.prod(_latent.shape[-1], device=self.config.device)
        elif what_latent == 'taskID':
            if taskID is None:
                raise ValueError('taskID should be provided')
            else:
                combined_input = torch.cat((input, taskID), dim=-1)
        
        if self.latent.shape != _latent.shape: # latent shape has changed. Reinit the latent and the optimizer to reset the pytorch computational graph to the new sizes
            # print('original latent shape', self.latent.shape)
            # print('Current latent shape', _latent.shape)
            self.latent = torch.nn.Parameter(_latent, requires_grad = True)
            # self.latent.data = self.latent.data.to(self.config.device)
            self.LU_optimizer = self.get_LU_optimizer()
        # else:
        #     self.latent.data = _latent.data.to(self.config.device)
        #     self.latent.requires_grad = True
        if self.config.predict_first_frame: # predict the first frame with initial zero input
            # also also importantly remove the last frame as it becomes unecessary. and need input to match len of latent
            zero_frame = torch.zeros_like(input[:, 0, ...])
            input = torch.cat((zero_frame.unsqueeze(1), input[:,:-1]), dim=1)

        combined_input = torch.cat((input, self.latent_activation_function(self.latent) ), dim=-1)

        return combined_input

    def update_latent(self, input, loss_function =None, logger = None, taskID=None, no_of_latent_steps = None):
        ''' updates the latent variable based on the input and the gradients
        will log the losses throughout optimization if logger is provided
        otherwise will return the loss of the first round before optimization occurs for comparison with loss after during learning..
        Provide no_of_latent_steps to override the config no_of_steps_in_latent_space
        '''
        if loss_function is None:
            loss_function = nn.MSELoss()
        for i in range(config.no_of_steps_in_latent_space if no_of_latent_steps is None else no_of_latent_steps):
            self.LU_optimizer.zero_grad()
            # self.latent.detach_() 
            # self.latent.requires_grad = True
            
            combined_input = self.combine_input_with_latent(input, what_latent='self')
            outputs, _ = self.forward(combined_input)
            outputs = torch.stack(outputs, dim=1)
            loss = loss_function(outputs, input) if self.config.predict_first_frame else loss_function(outputs, input[:, 1:, :])
            if logger is not None: 
                logger.log_updating_loss(loss.cpu().detach().numpy())
                logger.log_updating_latent(self.latent.clone().cpu().detach().numpy())
                logger.log_updating_combined_input(combined_input.clone().cpu().detach().numpy())
                logger.log_updating_output(outputs.cpu().detach().numpy())  
            if i ==0: before_optimization_loss = loss.cpu().detach().numpy() # only save the first round loss prior to optim
            loss = loss.sum() if self.config.loss_reduction_LU == 'sum' else loss.mean()
            loss.backward()
            self.adjust_latent_grads(self.config.latent_aggregation_op)
            self.LU_optimizer.step()
            # combined_input.detach_() # Cut the computational graph to save memory
        return before_optimization_loss 

    def adjust_latent_grads(self, apply_op = 'average'):
        ''' adjusts the gradients of the latent variable based on the config'''
        if apply_op == 'last':
            self.latent.grad = self.latent.grad[-1, ...]
        elif apply_op == 'first':
            self.latent.grad = self.latent.grad[0, ...]
        elif apply_op == 'average':
            # self.latent.grad = self.latent.grad.mean(dim=0)
            self.latent.grad = self.latent.grad.mean(dim=1).unsqueeze(1).expand_as(self.latent.grad).clone() 
            # note clone() is important because expand does not create a new tensor and then backward tries to write the grads on the next run but finds that all the grads are pointing to the same memory location
            
        elif apply_op == 'convolve':
            no_of_latent_chunks = self.config.latent_chuncks
            chunk_size = self.config.latent_dims[0] // no_of_latent_chunks
            for i in range(no_of_latent_chunks):
                convolution_window = self.config.latent_convolution_windows[i]
                convolution_kernel = torch.ones(1, convolution_window,1, device=self.config.device) / convolution_window
                self.latent.grad[:, i*chunk_size:(i+1)*chunk_size, :] = \
                np.convolve(self.latent.grad[:,:, i*chunk_size:(i+1)*chunk_size, ], convolution_kernel, mode='same')

            # the pytorch solution is not working
            # no_of_latent_chunks = self.config.latent_chuncks
            # chunk_size = self.config.latent_dims[0] // no_of_latent_chunks
            # for i in range(no_of_latent_chunks):
            #     convolution_window = self.config.latent_convolution_windows[i]
            #     convolution_kernel = torch.ones(1, convolution_window, 1, device=self.config.device) / convolution_window
            #     self.latent.grad[:, :, i*chunk_size:(i+1)*chunk_size ] = torch.conv1d(self.latent.grad[:, :, i*chunk_size:(i+1)*chunk_size], convolution_kernel)

        elif apply_op == 'none':
            pass

    def forward_generate(self, input, ):
        ''' loops through the input for seq_len//2 times and returns the hidden and cell states
        then uses the hidden state to predict the next frame and recursively generates seq_len//2 frames'''
        outputs = []
        batch_size = input.size(0)
        hidden_state = torch.zeros(batch_size, self.config.hidden_size, device=self.config.device)
        cell_state = torch.zeros(batch_size, self.config.hidden_size, device=self.config.device)
        input = self.input_layer(input)
        for i in range(self.config.no_of_frames_to_prompt_generate):
            hidden_state, cell_state = self.lstm_cell(input[:, i, ...], (hidden_state, cell_state))
            output = self.output_layer(hidden_state)
            outputs.append(output)

        input = output
        for i in range(self.config.seq_len - self.config.no_of_frames_to_prompt_generate - 1):
            input = self.input_layer(input)
            hidden_state, cell_state = self.lstm_cell(input, (hidden_state, cell_state))
            output = self.output_layer(hidden_state)
            outputs.append(output)
            input = output
        return hidden_state, cell_state, outputs

def run_model(logger, config, dataloader, model, criterion, epochs = None):
    training_losses_per_batch = []
    testing_loss = []
    training_losses_per_epoch = []

    epochs = config.epochs if epochs is None else epochs
    for epoch in range(epochs): # only two epochs for training with inference
        model.train()
        running_loss = 0.0
        total_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=total_batches)
        for bi, batch in pbar:
            inputs, batch_llcids, batch_hlcids = batch
            inputs = inputs.to(config.device)
            logger.log_input(inputs[:, -config.stride:, :].cpu().detach().numpy())
            model.WU_optimizer.zero_grad()
            model.LU_optimizer.zero_grad()

            model.reset_latent(batch_size = inputs.shape[0], seq_len = inputs.shape[1]) # pass input shape to re-init the latent in case dims have changed
            if config.no_of_steps_in_latent_space > 0 and config.allow_latent_updates:
                first_full_loss = model.update_latent(inputs, criterion, logger if bi == 0 else None) # log only the first batch to save memory

                logger.log_training_loss_before_latent_optimization(first_full_loss[:, -config.stride:, :])
            logger.log_latent_value(model.latent.clone()[:, -config.stride:, :].cpu().detach().numpy())

            if config.add_noise_to_input:
                inputs = inputs + torch.randn_like(inputs) * config.noise_std
            combined_input = model.combine_input_with_latent(inputs, what_latent='self')
            combined_input.detach_() # otherwise throws an error when backward pass is called

            logger.log_training_batch(combined_input[:, -config.stride:, :].cpu().detach().numpy())
            logger.llcids.append(batch_llcids[:, -config.stride:].cpu().detach().numpy())
            logger.hlcids.append(batch_hlcids[:, -config.stride:].cpu().detach().numpy())

            if config.no_of_steps_in_weight_space > 0:
                for i in range(config.no_of_steps_in_weight_space):
                    model.WU_optimizer.zero_grad()
                # Forward pass
                    outputs, _ = model(combined_input)
                    outputs = torch.stack(outputs, dim=1)
                # Compute the loss
                    loss = criterion(outputs, inputs) if config.predict_first_frame else criterion(outputs, inputs[:, 1:, :])
                    full_loss = loss.cpu().detach().numpy()
                    logger.log_training_loss(full_loss[:, -config.stride:, :])

                # Backward pass and optimization
                    loss = loss.sum() if config.loss_reduction_WU == 'sum' else loss.mean()
                    loss.backward()
                    model.WU_optimizer.step()
            else: # just run the model once to get the loss
                outputs, _ = model(combined_input)
                outputs = torch.stack(outputs, dim=1)
                loss = criterion(outputs, inputs) if config.predict_first_frame else criterion(outputs, inputs[:, 1:, :])
                logger.log_training_loss(loss.cpu().detach().numpy()[:, -config.stride:, :])
                loss = loss.sum() if config.loss_reduction_WU == 'sum' else loss.mean()
            logger.log_predicted_output(outputs.cpu().detach().numpy()[:, -config.stride:, :])

        # Update the running loss
            running_loss += loss.item()
            training_losses_per_batch.append(loss.item())

    # Print the average loss for the epoch
        average_loss = running_loss / len(dataloader)
        training_losses_per_epoch.append(average_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')


def test_model(model, dataloader_test, criterion, epoch = 0):
    # test the model
    model.eval()
    running_loss_test = 0.0
    for batch in dataloader_test:
        inputs, _, _ = batch
        inputs = inputs.to(config.device)

        # Forward pass
        combined_input = model.combine_input_with_latent(inputs, what_latent='init')
        outputs, _ = model(combined_input)
        outputs = torch.stack(outputs, dim=1)

        # Compute the loss
        # loss = criterion(outputs, inputs[:, 1:, :]).mean()
        loss = criterion(outputs, inputs) if config.predict_first_frame else criterion(outputs, inputs[:, 1:, :])
        loss = loss.mean()

        # Update the running loss
        running_loss_test += loss.item()

    # Print the average loss for the epoch
    average_loss_test = running_loss_test / len(dataloader_test)
    print(f'Epoch [{epoch+1}/{config.passive_epochs}], Test Loss: {average_loss_test:.4f}')
    return average_loss_test

#%% 
logger = Logger()
config = Config(experiment_name='contextual_switching_task')
# config = Config(experiment_name='META')
# config.stride = 50
# config.seq_len = 50 # to capture low level latent
# config.seq_len = 150 # to capture high level latent
# config.batch_size = 100
# # config.latent_aggregation_op = 'average' 
# config.latent_aggregation_op = 'none' 
# # config.latent_aggregation_op = 'convolve' # latent grads are chunked and convolved with a unique window for each chunk
# config.no_of_steps_in_weight_space = 1

if config.experiment_name == 'META':
    pass
else:
    # conditions_to_run = 'long_horizon_passive'
    conditions_to_run = 'long_horizon_active'
    # conditions_to_run = 'short_horizon_passive'
    # conditions_to_run = 'short_horizon_active'
    if conditions_to_run == 'long_horizon_passive':
        config.seq_len = 150
        config.passive_epochs = 0
        config.epochs = 1
        config.no_of_steps_in_latent_space = 5
        config.no_of_steps_in_weight_space = 0
        config.batch_size = 100 # does not need to be 1 because WU 0
    elif conditions_to_run == 'long_horizon_active':
        config.seq_len = 150
        config.passive_epochs = 1
        config.epochs = 1
        config.no_of_steps_in_latent_space = 5
        config.no_of_steps_in_weight_space = 1
        config.batch_size = 100 # not 1 just to avoid shape issues
        config.block_size = 200 
    elif conditions_to_run == 'short_horizon_passive':
        config.seq_len = 10
        config.passive_epochs = 0
        config.epochs = 1
        config.no_of_steps_in_latent_space = 5
        config.no_of_steps_in_weight_space = 0
        config.batch_size = 100
    elif conditions_to_run == 'short_horizon_active':
        config.seq_len = 5
        config.passive_epochs = 0
        config.epochs = 1
        config.no_of_steps_in_latent_space = 5
        config.no_of_steps_in_weight_space = 1
        config.batch_size = 20

if config.experiment_name == 'META':
    # Meta dataset loader parameters
    np.random.seed(0)
    min_len = 30
    holdout_llcid = 28
    exlude_multi_chapter = 1
    verbose = 0
    meta = METAVideos(holdout_llcid=holdout_llcid, min_len=min_len, exlude_multi_chapter=exlude_multi_chapter, verbose=verbose)
    no_of_videos_to_use = 0
    dataset = METAdataset(meta, config, no_of_videos_to_use=no_of_videos_to_use)
    dataset_test = METAdataset(meta, config, no_of_videos_to_use=no_of_videos_to_use, split='test')
else:
    dataset = TaskDataset(config.no_of_blocks, config)
    dataset_test = TaskDataset(config.no_of_blocks, config)

# Create dataset
# dataset = METAdataset(dataset, config, no_of_videos_to_use=0)
# Create dataloader
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
# The dataloader will return a list of three elements. 
# First element is a tensor of shape ( batch_size, seq_len, 30)
# Second element is a tensor of shape ( batch_size, seq_len) with llc ids
# Third element is a tensor of shape ( batch_size, seq_len) with hlc ids

#%%
# check if the path exists
if not os.path.exists(config.export_path):
    os.makedirs(config.export_path)
save_path = f'{config.export_path}{config.experiment_name}_LUs_{config.no_of_steps_in_latent_space}_batch_size{config.batch_size}'

model = RNN_with_latent(config).to(config.device)
criterion = nn.MSELoss(reduction='none')
#%% RUN THE MODEL
print('Running the model')
print('with configs: ')
print(config.__dict__)
if config.load_saved_model:
    model = torch.load(f'{save_path}_model.pth')
    logger = torch.load(f'{save_path}_logger.pth')
    print(f'Model loaded from {save_path}')
else:
    config.allow_latent_updates = False
    run_model(logger, config, dataloader, model, criterion, epochs = config.passive_epochs)
    logger = Logger() # reset the logger
    config.allow_latent_updates = True
    run_model(logger, config, dataloader, model, criterion)

# save
if config.save_model:
    torch.save(model, (f'{save_path}_model.pth'))
    torch.save(logger, (f'{save_path}_logger.pth'))
    print(f'Model saved to {save_path}')

#%%!
fig = plot_behavior(explore_data_container, logger, config, print_shapes=False)#, x2=3000)
fig.savefig(f'{config.export_path}{config.experiment_name}_behavior.pdf', format='pdf', bbox_inches='tight')
plot_tnse_previous_colors(logger, config)
plt.savefig(f'{config.export_path}{config.experiment_name}_tsne.pdf', format='pdf', bbox_inches='tight')
#plot_tnse(logger, config)
#%% Generate the latent GIF

if False: # Change to true to get a long seq latent GIF
    config.seq_len = 500
    config.latent_aggregation_op = 'none'
    model.config.latent_aggregation_op = config.latent_aggregation_op
it = (iter(dataloader))
inputs, _, _ = next(it)
inputs, _, _ = next(it)
inputs = inputs.to(config.device)
# for example_batch_no in [0, 50, 99]:
example_batch_no = 0
filename = f'{save_path}_latent_optimization_agg_batch_{example_batch_no}'
demo_latent_plots_with_outputs(model, inputs, filename, criterion, config, LU_steps=100, example_batch_no = example_batch_no)
# load and show GIF
from IPython.display import Image
Image(filename=filename+f'.gif')

# %%
# save the logger object using numpy 
# np.save('logger.npy', logger)
#%% Load the logger object
# logger = np.load('./../../logger.npy', allow_pickle=True).item()
#%%
plot_tnse(logger, config)

#%%  Predict %% Now trying to get inference:
logger = Logger()
config.stride = 1
config.seq_len = 10 #30 # to capture low level latent
# config.seq_len = 150 # to capture high level latent
# config.batch_size = 200
# config.latent_aggregation_op = 'none'
config.latent_aggregation_op = 'average'
model.config.latent_aggregation_op = config.latent_aggregation_op
# config.no_of_steps_in_latent_space = 100
# model.config.no_of_steps_in_latent_space = config.no_of_steps_in_latent_space
config.no_of_steps_in_weight_space = 0

# Create dataloader
dataset = TaskDataset(config.no_of_blocks, config)
dataset_test = TaskDataset(config.no_of_blocks, config)

dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, )
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, )

def predict(logger, config, dataloader, model, criterion, epochs = 1):
    training_losses_per_batch = []
    testing_loss = []
    training_losses_per_epoch = []

    for epoch in range(epochs): # only two epochs for training with inference
        model.train()
        running_loss = 0.0
        total_batches = min(config.limited_testing_samples_no, len(dataloader)) ######## NOTE LIMITED test
        # for bi, batch in tqdm(enumerate(dataloader)):
        pbar = tqdm(enumerate(dataloader), total=total_batches)
        for bi, batch in pbar:
            if bi > total_batches:
                break
            inputs, batch_llcids, batch_hlcids = batch
            logger.log_input(inputs[:, -config.stride:, :].cpu().detach().numpy())
            inputs = inputs.to(config.device)
            # get the first prediction
            combined_input = model.combine_input_with_latent(inputs, what_latent='self')
            # combined_input.detach_() # otherwise throws an error when backward pass is called
            outputs, _ = model(combined_input)
            outputs = torch.stack(outputs, dim=1)
            loss = criterion(outputs, inputs) if config.predict_first_frame else criterion(outputs, inputs[:, 1:, :])
            logger.log_prediction_loss(loss.cpu().detach().numpy()[:, -config.stride:, :])
            loss = loss.mean()
            logger.log_predicted_output(outputs.cpu().detach().numpy()[:, -config.stride:, :])

            logger.log_training_batch(combined_input[:, -config.stride:, :].cpu().detach().numpy())
            logger.llcids.append(batch_llcids[:, -config.stride:].cpu().detach().numpy())
            logger.hlcids.append(batch_hlcids[:, -config.stride:].cpu().detach().numpy())

            # now update the latent based on the new input
            model.WU_optimizer.zero_grad()
            model.LU_optimizer.zero_grad()
            
            # Logging before latent update so that input, output, and the corrosponding latent align
            logger.log_latent_value(model.latent.clone()[:, -config.stride:, :].cpu().detach().numpy())

            model.reset_latent(batch_size = inputs.shape[0], seq_len = inputs.shape[1]) # pass input shape to re-init the latent in case dims have changed
            if config.no_of_steps_in_latent_space > 0:
                first_full_loss = model.update_latent(inputs, criterion,  logger if bi == 0 else None)
                logger.log_training_loss_before_latent_optimization(first_full_loss[:, -config.stride:, :])



            if config.no_of_steps_in_weight_space > 0:
                raise ValueError('No weight space optimization during inference')
                for i in range(config.no_of_steps_in_weight_space):
                    model.WU_optimizer.zero_grad()
                # Forward pass
                    outputs, _ = model(combined_input)
                    outputs = torch.stack(outputs, dim=1)
                # Compute the loss
                    loss = criterion(outputs, inputs) if config.predict_first_frame else criterion(outputs, inputs[:, 1:, :])
                    full_loss = loss.cpu().detach().numpy()
                    logger.log_training_loss(full_loss[:, -config.stride:, :])

                # Backward pass and optimization
                    loss = loss.sum() if config.loss_reduction_WU == 'sum' else loss.mean()
                    loss.backward()
                    model.WU_optimizer.step()

        # Update the running loss
            running_loss += loss.item()
            training_losses_per_batch.append(loss.item())

    # Print the average loss for the epoch
        average_loss = running_loss / len(dataloader)
        training_losses_per_epoch.append(average_loss)
        print(f'Epoch [{epoch+1}/{config.epochs}], Loss: {average_loss:.4f}')
 
model.reset_latent(batch_size = config.batch_size, seq_len = config.seq_len)
predict(logger, config, dataloader_test, model, criterion)
_ = plot_behavior(explore_data_container, logger, config, x2=None)
plot_tnse(logger, config)


#%%
# dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
# inputs, _, _ = next(iter(dataloader_test))
# config.latent_aggregation_op = 'average'
config.seq_len = 500
config.latent_aggregation_op = 'none'
model.config.latent_aggregation_op = config.latent_aggregation_op

dataset_temp = TaskDataset(config.no_of_blocks, config)
dataloader_temp = DataLoader(dataset_temp, batch_size=config.batch_size, shuffle=True, )

inputs, _, _ = next(iter(dataloader))
inputs = inputs.to(config.device)

# model.reset_latent(batch_size=inputs.shape[0], seq_len=inputs.shape[1])
# demo_latent_plots(model, inputs, 'latent_optimization_meta_no_agg', criterion, LU_steps=50)
filename = f'{save_path}_latent_optimization_meta_no_agg'
demo_latent_plots_with_outputs(model, inputs, filename, criterion, config, LU_steps=100)
# load and show GIF
from IPython.display import Image
Image(filename=filename+'.gif')


#%%  META hold out test
if config.experiment_name == 'META':
    logger = Logger()
    dataset_holdout = METAdataset(meta, config, no_of_videos_to_use=0, split='training', get_holdout=True)
    dataloader_holdout = DataLoader(dataset_holdout, batch_size=config.batch_size, shuffle=False, )
    model.reset_latent(batch_size = config.batch_size, seq_len = config.seq_len)
    predict(logger, config, dataloader_holdout, model, criterion)

    fig = plot_behavior(explore_data_container, logger, config, x2=None)
    fig.tight_layout()
plt.show()
# %%
## Print out all opened figures to a PDF
# from matplotlib.backends.backend_pdf import PdfPages
# pdf = PdfPages(f'./{config.export_path}_noise_{config.add_noise_to_input}_figures.pdf')
# figs = [plt.figure(n) for n in plt.get_fignums()]
# for fig in figs:
#     pdf.savefig(fig)
# pdf.close()

# %%
