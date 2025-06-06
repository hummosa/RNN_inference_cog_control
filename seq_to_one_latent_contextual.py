# %% 
auto_reload = True
if auto_reload:
    #!%load_ext autoreload
    #!%autoreload 2
    pass
#%%
''' This file will focus on optimizing one latent for an entire sequence
The contextual switching task worked. 
Using a seq_len of about 20 or 30 tunes the latent to the low level context.
Using a seq_len of about 150 tunes the latent to the high level context.

This was more evident in very long runs with about 500 blocks.
Reducing block size seems to pick up on the high level latent even 
with short seq_len.

'''
import torch
from torch.utils.data import Dataset, DataLoader   
import torch.nn as nn

import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.cluster.hierarchy import dendrogram, linkage

# sns.set(style='white', palette='colorblind', context='talk')
# cpal = sns.color_palette('colorblind')

from torch.utils.data import Dataset, DataLoader
import numpy as np
from batch_hiflat_visualization import *

class Config:
    def __init__(self, experiment_name = 'contextual_switching_task'):
        if experiment_name == 'contextual_switching_task':
            self.input_size = 1 # gets updated below
            self.hidden_size = 32
            self.output_size = 1
            self.seq_len = 10
            self.stride = 1 # self.seq_len
            self.batch_size = 100
        elif experiment_name == 'META':
            self.input_size = 30 # gets updated below
            self.hidden_size = 200
            self.output_size = 30
            self.seq_len = 50
            self.stride = self.seq_len

        self.WU_lr = 0.001
        self.no_of_frames_to_prompt_generate = 10
        self.predict_first_frame = True

        # latent
        self.LU_lr = 0.01
        self.latent_type = '1d_latent'
        self.latent_dims = [2] 
        self.latent_chuncks = 2 # how many chuncks to divide the latent into.
        self.latent_activation = 'none' #'softmax'
        self.latent_activation_dim = 0
        self.latent_activation_temp = 1 # not used for sigmoid
        self.momentum = 0.9 # only for sgd
        self.l2_loss = 0#0.0001 # weight decay for Adam
        self.LU_optimizer = 'Adam' # 'SGD' 'Adam'
        self.input_size += np.prod(self.latent_dims) # update the input size to include the latent
        self.loss_reduction_LU = 'mean' # 'sum' 'mean' 'none'
        self.loss_reduction_WU = 'mean' # 'mean' 'none'

        # latent config for latent I and II [I, II]
        self.latent_convolution_windows = [5, 20] # smootheness of each latent chunk
        self.latent_value_to_use = ['last', 'first'] # 'average' 'filtered' # updates the latent value witht the grad of the first or last element of the sequence.
        self.no_of_steps_in_latent_space = 10
        self.no_of_steps_in_weight_space = 1
        self.latent_aggregation_op = 'none' # 'convolve' 'last' 'first' 'average' 'none' # how to aggregate the gradients of the latent over the sequence

        # Task parameters
        # Contexutal task
        self.no_of_blocks = 50
        self.block_size = 50
        self.latent_change_interval = 1
        self.high_level_latent_change_interval_in_blocks = 3
        self.default_std = 0.1

        # training
        self.epochs = 1
        self.passive_epochs = 1
        self.test_split = 0.2

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#%%
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

    def update_latent(self, input, loss_function =None, taskID=None, logger = None):
        ''' updates the latent variable based on the input and the gradients
        will log the losses throughout optimization if logger is provided
        otherwise will return the loss of the first round before optimization occurs for comparison with loss after during learning..
        '''
        if loss_function is None:
            loss_function = nn.MSELoss()
        for i in range(self.config.no_of_steps_in_latent_space):
            self.LU_optimizer.zero_grad()
            combined_input = self.combine_input_with_latent(input, what_latent='self')
            outputs, _ = self.forward(combined_input)
            outputs = torch.stack(outputs, dim=1)
            loss = loss_function(outputs, input) if self.config.predict_first_frame else loss_function(outputs, input[:, 1:, :])
            if logger is not None: 
                logger.log_updating_loss(loss.cpu().detach().numpy())
                logger.log_updating_latent(self.latent.cpu().detach().numpy())
                logger.log_updating_combined_input(combined_input.cpu().detach().numpy())
            if i ==0: before_optimization_loss = loss.cpu().detach().numpy() # only save the first round loss prior to optim
            loss = loss.sum() if self.config.loss_reduction_LU == 'sum' else loss.mean()
            loss.backward()
            self.adjust_latent_grads(self.config.latent_aggregation_op)
            self.LU_optimizer.step()
        return before_optimization_loss 

    def adjust_latent_grads(self, apply_op = 'average'):
        ''' adjusts the gradients of the latent variable based on the config'''
        if apply_op == 'last':
            self.latent.grad = self.latent.grad[-1, ...]
        elif apply_op == 'first':
            self.latent.grad = self.latent.grad[0, ...]
        elif apply_op == 'average':
            # self.latent.grad = self.latent.grad.mean(dim=0)
            self.latent.grad = self.latent.grad.mean(dim=1).unsqueeze(1).expand_as(self.latent.grad)
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

def run_model(logger, config, dataloader, model, criterion, epochs = 1):
    training_losses_per_batch = []
    testing_loss = []
    training_losses_per_epoch = []

    for epoch in range(epochs): # only two epochs for training with inference
        model.train()
        running_loss = 0.0
        for bi, batch in tqdm(enumerate(dataloader)):
            inputs, batch_llcids, batch_hlcids = batch
            inputs = inputs.to(config.device)
            logger.log_input(inputs[:, -config.stride:, :].cpu().detach().numpy())
            model.WU_optimizer.zero_grad()
            model.LU_optimizer.zero_grad()

            # model.latent.detach_() # avoids error saying some write to variables point to a single memory location
            #unsupported operation: more than one elemen
            # model.latent.requires_grad = True
            model.init_latent(batch_size = inputs.shape[0], seq_len = inputs.shape[1])
            model.reset_latent(batch_size = inputs.shape[0], seq_len = inputs.shape[1]) # pass input shape to re-init the latent in case dims have changed
            if config.no_of_steps_in_latent_space > 0:
                first_full_loss = model.update_latent(inputs, criterion, logger)

                logger.log_training_loss_before_latent_optimization(first_full_loss[:, -config.stride:, :])
            logger.log_latent_value(model.latent[:, -config.stride:, :].cpu().detach().numpy())

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
        print(f'Epoch [{epoch+1}/{config.epochs}], Loss: {average_loss:.4f}')


def test_model(model, dataloader_test, criterion, epoch = 0):
    # test the model
    model.eval()
    running_loss_test = 0.0
    for batch in dataloader_test:
        inputs, _, _ = batch
        inputs = inputs.to(config.device)

        # Forward pass
        combined_input = model.combine_input_with_latent(inputs, what_latent='zeros')
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



# %%
# Create dataset
logger = Logger()
config = Config()
config.stride = 1
config.seq_len = 10 # to capture low level latent
# config.seq_len = 150 # to capture high level latent
config.batch_size = 2
config.latent_aggregation_op = 'average' 
# config.latent_aggregation_op = 'none' 
# config.latent_aggregation_op = 'convolve' # latent grads are chunked and convolved with a unique window for each chunk
config.no_of_steps_in_latent_space = 1
config.no_of_steps_in_weight_space = 1

# Create dataset
config.no_of_blocks = 50
config.block_size = 50

dataset = TaskDataset(config.no_of_blocks, config)
dataset_test = TaskDataset(config.no_of_blocks, config)

# Cannot visualize data with stride 1
# plot_dataset_sample(explore_data_container, config, dataset)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, )
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, )
# The dataloader will return a list of three elements. 
# First element is a tensor of shape ( batch_size, seq_len, 1) with data values
# Second element is a tensor of shape ( batch_size, seq_len, 1) with llc values
# Third element is a tensor of shape ( batch_size, seq_len, 1) with hlc values

#%% Train the RNN using traditional backpropagation for passive epochs
model = RNN_with_latent(config).to(config.device)

criterion = nn.MSELoss(reduction='none')

# TRAIN with stride 1

run_model(logger, config, dataloader, model, criterion)
#%%
fig = plot_behavior(explore_data_container, logger, config,)



#%%

plot_tnse(logger, config)


#%%
# try:

# for bi, batch in enumerate(dataloader):
#     inputs, batch_llcids, batch_hlcids = batch
#     inputs = inputs.to(config.device)
#     break
# uncomment to save the gif
# demo_latent_plots(model, inputs, 'latent_optimization_context_task')

# except:
# print('failed to create the gif')


# %% Now trying to get inference:
logger = Logger()
config.stride = 1
# config.seq_len = 30 # to capture low level latent
# config.seq_len = 150 # to capture high level latent
# config.batch_size = 200
config.latent_aggregation_op = 'none'
# config.no_of_steps_in_latent_space = 0
config.no_of_steps_in_weight_space = 0

# Create dataloader
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, )
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, )

#%%
# run_model(logger, config, dataloader, model, criterion)

#%%  Predict

def predict(logger, config, dataloader, model, criterion, epochs = 1):
    training_losses_per_batch = []
    testing_loss = []
    training_losses_per_epoch = []

    for epoch in range(epochs): # only two epochs for training with inference
        model.train()
        running_loss = 0.0
        for bi, batch in tqdm(enumerate(dataloader)):
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

            model.reset_latent(batch_size = inputs.shape[0], seq_len = inputs.shape[1]) # pass input shape to re-init the latent in case dims have changed
            if config.no_of_steps_in_latent_space > 0:
                first_full_loss = model.update_latent(inputs, criterion, logger)
                logger.log_training_loss_before_latent_optimization(first_full_loss[:, -config.stride:, :])

            logger.log_latent_value(model.latent[:, -config.stride:, :].cpu().detach().numpy())


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
predict(logger, config, dataloader, model, criterion)

_ = plot_behavior(explore_data_container, logger, config, x2=None)
plt.show()

# %% parameter search

def train_and_test_loop(Config, RNN_with_latent, run_model, Logger, dataset, dataset_test):
    logger = Logger()
    config = Config(experiment_name = 'contextual_switching_task')
    #
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, )
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, )

    model = RNN_with_latent(config).to(config.device)
    criterion = nn.MSELoss(reduction='none')

    run_model(logger, config, dataloader, model, criterion)

    fig_training = plot_behavior(explore_data_container, logger, config)

    logger = Logger()
    config.no_of_steps_in_weight_space = 0
    predict(logger, config, dataloader, model, criterion)

    fig_testing = plot_behavior(explore_data_container, logger, config, x2=None)
    # plt.show()
    return fig_training, fig_testing

# train_and_test_loop(Config, RNN_with_latent, run_model, Logger, dataset, dataset_test)
#%%

export_folder = './exports/'
import os
if not os.path.exists(export_folder):
    os.makedirs(export_folder)

def sweep_parameters(config, parameters_to_sweep, parameter_values):
    if len(parameters_to_sweep) != 3:
        raise ValueError("Please provide a list of three strings for parameters_to_sweep.")
    
    for param1 in parameter_values[0]:
        for param2 in parameter_values[1]:
            for param3 in parameter_values[2]:
                setattr(config, parameters_to_sweep[0], param1)
                setattr(config, parameters_to_sweep[1], param2)
                setattr(config, parameters_to_sweep[2], param3)
                print(f'Running with {parameters_to_sweep[0]} = {param1}, {parameters_to_sweep[1]} = {param2}, {parameters_to_sweep[2]} = {param3}')
                fig_training, fig_testing = train_and_test_loop(Config, RNN_with_latent, run_model, Logger, dataset, dataset_test)
                fig_training.savefig(f'{export_folder}{parameters_to_sweep[0]}_{param1}_{parameters_to_sweep[1]}_{param2}_{parameters_to_sweep[2]}_{param3}_training.png')
                fig_testing.savefig(f'{export_folder}{parameters_to_sweep[0]}_{param1}_{parameters_to_sweep[1]}_{param2}_{parameters_to_sweep[2]}_{param3}_testing.png')
                plt.close('all')

parameters_to_sweep = ['no_of_steps_in_latent_space', 'batch_size', 'LU_lr']
parameter_values = [
    [1, 10, 100],  # Values for no_of_steps_in_latent_space
    [1, 5],       # Values for batch_size
    [0.1, 0.01]       # Values for LU_lr
]

sweep_parameters(config, parameters_to_sweep, parameter_values)


