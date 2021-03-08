# Created by Yiran Wu @ Feb 3, 2021, all rights reserved
# Email: yrwu@ucdavis.edu
# This convLSTM referenced the following two git:
#   https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
#   https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py

from config import *
'''
config.py includes imported frameworks and global variables
env global: VERSION[string], USE_CUDA[boolean], DEVICE[torch.device]
hyper params[int]:  BATCH_SIZE, TIME_STEP, HEIGHT, WIDTH
hyper params[list]: LAYERS

If this file is used along, import the following
import numpy as np
import torch
import torch.nn as nn
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(DEVICE) if USE_CUDA else autograd.Variable(*args, **kwargs)
'''



############################################################
#################ConvLSTMCell###############################
class ConvLSTMCell(nn.Module):
    """This is a peekhole ConvLSTM Cell.
    
    This class inherits PyTorch nn.Module.
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size, height, width):
        """
        !Caution: to maintain same padding, odd kernel_size should always be used
        
        Args:
            input_channels(int): Define input channels' size.
            hidden_channels(int): Define hidden
            kernel_size(tuple or int): Define kernel_size of self.convW. Notice !Caution.
            height(int): The height of the image.
            width(int): The width of the image.
    
        Attributes:
            Partly described in Args above.
            Others descriptions see inline comments.
        """
        super().__init__()
        
        if(isinstance(kernel_size, int)):
            kernel_size = (kernel_size,kernel_size)
        
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # to maintain same padding
        
        # this is equivalent to combination of weights Wxi,Whi,  Wxf,Whf,  Wxc,Whc,   Wxo,Who
        # the input of the conv layer will be combine channel-wise
        # the ouput would be 4 times hidden_channels, with each represent two weight above, as listed
        # stride = 1, bias=True
        self.convW = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 4, kernel_size, 1, self.padding, bias=True)
        
        # cell state weight / peekhole weight
        self.Wci = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width))
        self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width))
        self.Wco = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width))
        
    
    def forward(self, x, h, c):
        """ LSTM caculations.
        
        Implement the forward method inherited from nn.Module.
        
        Args:
            x(torch.Tensor): curr input,  with shape (batch_size, input_channels, height, width)
            h(torch.Tensor): hidden state from t-1, with shape (batch_size, hidden_channels, height, width)
            c(torch.Tensor): cell state from t-1,  with shape (batch_size, hidden_channels, height, width)
        """

        hx = torch.cat([x, h], dim=1) # h+x combined along channel
        del x,h # free memory immediately
        
        y = self.convW(hx) # mutiply weight to input -> conv
        del hx # free memory immediately
        
        yi, yf, yc, yo = torch.split(y, self.hidden_channels, dim=1) # split to each temp var
        del y # free memory immediately
        
        # caculation
        i_t = torch.sigmoid(yi + self.Wci * c)
        f_t = torch.sigmoid(yf + self.Wcf * c)
        C_t = f_t * c + i_t * torch.tanh(yc)
        del i_t,f_t, yi, yf, yc # free memory immediately
        o_t = torch.sigmoid(yo + self.Wco * c)
        H_t = o_t * torch.tanh(C_t)
        return H_t, C_t
        

############################################################
####################ConvLSTMLayer###########################
class ConvLSTMLayer(nn.Module):
    '''This a convLSTM Layer, which is composed of several convLSTMcells.
    
    Notice:
        - Each convLSTMcell will accept input, hidden state and cell state with same dimension, from continuous time steps,
        and output cell states and hidden states with the same dimension, which would be input into the next cell in the layer.
        - Same kernel_size for all cells in one layer.
    '''
    def __init__(self, input_channels, hidden_channels, kernel_size, time_steps, height, width):
        """
        Args:
            input_channels(int): channel dimesion of input for all cells in this layer
            hidden_channels(int): equivalent to ouput channel, specify both the channel of both hidden state and cell state
            kernel_size(int/tuple): same kernel_size for all cells in one layer.
            time_steps(int): describe how many time steps are used to predict.
            height(int), width(int): same height, width for all cells in this layer
        
        Attributes:
            Partly described in Args above.
            Others descriptions see inline comments.
        """
        super().__init__()
        
        # int to tuple
        if(isinstance(kernel_size, int)):
            kernel_size = (kernel_size,kernel_size)
            
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.time_steps = time_steps # fixed input time steps
        self.height = height # fix h, w of input
        self.width = width
        
        
        cell_list = []
        for i in range(time_steps):
            cell_list.append(ConvLSTMCell(input_channels, hidden_channels, kernel_size, height, width))
        self.cell_list = nn.ModuleList(cell_list)
        """cell_list(nn.ModuleList): Append size=time_steps of ConvLSTMCell to form a layer """
        
        
    def forward(self, x):
        """Implement the forward method inherited from nn.Module.
        
        Notice:
            The cell states are not returned here, but under some circumstances it is useful.
        
        Args:
            x(torch.Tensor): input with dimension (batch_size, time, channels, height, width)
            
        Returns:
            output_list(list of torch.Tensor): contains all outputs from this layer.
            
        Raises:
            AssertionError: see keyword assert.
        """
        
        batch_size, time_steps, _, height, width = x.shape
        
        # assert equals
        assert time_steps == self.time_steps, f"Except time step {self.time_steps}, but got {time_steps}"
        assert height == self.height, f"Except height {self.height}, but got {height}"
        assert width == self.width, f"Except width {self.width}, but got {width}"
        
        # init c,h to zeros
        h = Variable(torch.zeros(batch_size, self.hidden_channels, height, width))
        c = Variable(torch.zeros(batch_size, self.hidden_channels, height, width))
       
        # forward
        output_list = []
        for i in range(time_steps):
            h, c = self.cell_list[i](x[:,i,:,:,:], h, c)
            output_list.append(h)

        return output_list

  
  
  
############################################################
######################ConvLSTM##############################

def _validate_kernel(kernel_size, num_layers):
    '''To validate and transform input for initializations.
    
    Intended to be used only in the following ConvLSTM class.
    
    Arges:
        kernel_size(tuple or int): the kernel_size to be transformed.
        num_layers(int): num of layers, decides length of the output list.
        
    Returns:
        kernel_size(list of tuple):
    
    '''
    if(isinstance(kernel_size, int)):
        kernel_size = (kernel_size,kernel_size)
        kernel_size = [kernel_size] * num_layers
        
    if(isinstance(kernel_size, tuple)):
        kernel_size = [kernel_size] * num_layers
        
    assert len(kernel_size)== num_layers, 'Kernel size mismatched!'
    return kernel_size


class ConvLSTM(nn.Module):
    '''A ConvLSTM class.
    
    ConvLSTM stacked one or several ConvLSTMLayers, and the ouput would be the ouput from the last layer.
    This is a stateless ConvLSTM : hidden state from previous batch will not be input into next batch as initial hidden state

    '''

    def __init__(self, channels, kernel_size, time_steps, height, width):
        '''
        Notice:
            - channels is a list that defined the input and output channel of each layer
            Example: if channels = [256, 256], that means this ConvLSTM has one layer, with input channel 256 and ouput channel 256
            if channels = [256, 512, 256], that means this ConvLSTM has two layers, the first layer has input channel 256 and ouput channel 512, the second layer has input channel 512 and ouput channel 256.
        
            - kernel_size can be tuple/int/list, it gives more flexibility to the structure, also len(kernel_size) == len(channels) - 1 == num_layers.
                if kernel_size is tuple/int, _validate_kernel will put kernel_size in list, meaning each layer has the same kernel_size.
                if kernel_size is list, it should be list of tuple, assert len(kernel_size) == len(channels)-1
        
        Args:
            channels(list): discribe channel dimensions and num of layers
            kernel_size(tuple/int/list): specify ouput channels of each layer
            time_steps(int): describe how many time steps are used to predict.
            height(int), width(int): all layers have uniform height, width for now
            
        Attributes:
            Partly is same as Args above.
            Others descriptions see inline comments.
        '''
        super().__init__()
        
        self.channels = channels
        self.num_layers = len(channels) - 1
        self.kernel_size = _validate_kernel(kernel_size, self.num_layers)
        
        
        # initialize Layers
        layers = []
        for i in range(self.num_layers):
            layers.append(ConvLSTMLayer(channels[i], channels[i+1], self.kernel_size[i], time_steps, height, width))
        self.layers = nn.ModuleList(layers)
        
    
    
    def forward(self, input):
        """Implement the forward method inherited from nn.Module.
        
        Notice:
            This ConvLSTM is designed to return only outputs from the last layer, but ouput from all layers could be accessed if needed.
            The cell states are also not returned here, also see comments in class ConvLSTMLayer->forward
        
        Args:
            input(torch.Tensor): input with dimension (batch_size, time, channels, height, width)
            
        Returns:
            h(list of torch.Tensor): contains all outputs from the last layer.
            
        Raises:
            AssertionError: see keyword assert.
        """
        # batch_size, time_steps, channels, height, weight = input.shape
        
        h = input
        for i in range(self.num_layers):
            h = self.layers[i](h) # h is a list of tensor, len(h) = time_steps, each tensor has dim (batch_size, 1, channels, height, width)
            h = torch.stack(h, dim=1) # stack all time steps, get torch.Tensor (batch_size, time_steps, channels, height, width)
        return h
          
  
  
'''
# USE FOR TEST:

lstm = ConvLSTM(channels= [10, 20, 20],
                      kernel_size = 3,
                      time_steps = 4,
                      height= 5,
                      width = 5)

x = torch.rand(8, 4, 10, 5, 5) #(batch_size, time, channels, height, width)

o,c = lstm(x)
print(o[0].shape)
print(len(o))
        
'''
