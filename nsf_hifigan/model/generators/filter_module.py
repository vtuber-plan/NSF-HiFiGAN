
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from nsf_hifigan.model.modules.conv1_keep_length import Conv1dKeepLength

class TimeVarFIRFilter(nn.Module):
    """ TimeVarFIRFilter
    Given sequences of filter coefficients and a signal, do filtering
    
    Filter coefs: (batchsize=1, signal_length, filter_order = K)
    Signal:       (batchsize=1, signal_length, 1)
    
    For batch 0:
     For n in [1, sequence_length):
       output(0, n, 1) = \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)
       
    Note: filter coef (0, n, :) is only used to compute the output 
          at (0, n, 1)
    """
    def __init__(self):
        super(TimeVarFIRFilter, self).__init__()
    
    def forward(self, signal, f_coef):
        """ 
        Filter coefs: (batchsize=1, signal_length, filter_order = K)
        Signal:       (batchsize=1, signal_length, 1)
        
        Output:       (batchsize=1, signal_length, 1)
        
        For n in [1, sequence_length):
          output(0, n, 1)= \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)
          
        This method may be not efficient:
        
        Suppose signal [x_1, ..., x_N], filter [a_1, ..., a_K]
        output         [y_1, y_2, y_3, ..., y_N, *, * ... *]
               = a_1 * [x_1, x_2, x_3, ..., x_N,   0, ...,   0]
               + a_2 * [  0, x_1, x_2, x_3, ..., x_N,   0, ...,  0]
               + a_3 * [  0,   0, x_1, x_2, x_3, ..., x_N, 0, ...,  0]
        """
        signal_l = signal.shape[1]
        order_k = f_coef.shape[-1]

        # pad to (batchsize=1, signal_length + filter_order-1, dim)
        padded_signal = F.pad(signal, (0, 0, 0, order_k - 1))
        
        y = torch.zeros_like(signal)
        # roll and weighted sum, only take [0:signal_length]
        for k in range(order_k):
            y += torch.roll(padded_signal, k, dims=1)[:, 0:signal_l, :] \
                      * f_coef[:, :, k:k+1]
        # done
        return y


class SincFilter(nn.Module):
    """ SincFilter
        Given the cut-off-frequency, produce the low-pass and high-pass
        windowed-sinc-filters.
        If input cut-off-frequency is (batchsize=1, signal_length, 1),
        output filter coef is (batchsize=1, signal_length, filter_order).
        For each time step in [1, signal_length), we calculate one
        filter for low-pass sinc filter and another for high-pass filter.
        
        Example:
        import scipy
        import scipy.signal
        import numpy as np
        
        filter_order = 31
        cut_f = 0.2
        sinc_layer = SincFilter(filter_order)
        lp_coef, hp_coef = sinc_layer(torch.ones(1, 10, 1) * cut_f)
        
        w, h1 = scipy.signal.freqz(lp_coef[0, 0, :].numpy(), [1])
        w, h2 = scipy.signal.freqz(hp_coef[0, 0, :].numpy(), [1])
        plt.plot(w, 20*np.log10(np.abs(h1)))
        plt.plot(w, 20*np.log10(np.abs(h2)))
        plt.plot([cut_f * np.pi, cut_f * np.pi], [-100, 0])
    """
    def __init__(self, filter_order):
        super(SincFilter, self).__init__()
        # Make the filter oder an odd number
        #  [-(M-1)/2, ... 0, (M-1)/2]
        # 
        self.half_k = (filter_order - 1) // 2
        self.order = self.half_k * 2 +1
        
    def hamming_w(self, n_index):
        """ prepare hamming window for each time step
        n_index (batchsize=1, signal_length, filter_order)
            For each time step, n_index will be [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 0, :] = [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 1, :] = [-(M-1)/2, ... 0, (M-1)/2]
            ...
        output  (batchsize=1, signal_length, filter_order)
            output[0, 0, :] = hamming_window
            output[0, 1, :] = hamming_window
            ...
        """
        # Hamming window
        return 0.54 + 0.46 * torch.cos(2 * np.pi * n_index / self.order)
    
    def sinc(self, x):
        """ Normalized sinc-filter sin( pi * x) / pi * x
        https://en.wikipedia.org/wiki/Sinc_function
        
        Assume x (batchsize, signal_length, filter_order) and 
        x[0, 0, :] = [-half_order, - half_order+1, ... 0, ..., half_order]
        x[:, :, self.half_order] -> time index = 0, sinc(0)=1
        """
        y = torch.zeros_like(x)
        y[:,:,0:self.half_k]=torch.sin(np.pi * x[:, :, 0:self.half_k]) / (np.pi * x[:, :, 0:self.half_k])
        y[:,:,self.half_k+1:]=torch.sin(np.pi * x[:, :, self.half_k+1:]) / (np.pi * x[:, :, self.half_k+1:])
        y[:,:,self.half_k] = 1
        return y
        
    def forward(self, cut_f):
        """ lp_coef, hp_coef = forward(self, cut_f)
        cut-off frequency cut_f (batchsize=1, length, dim = 1)
    
        lp_coef: low-pass filter coefs  (batchsize, length, filter_order)
        hp_coef: high-pass filter coefs (batchsize, length, filter_order)
        """
        # create the filter order index
        with torch.no_grad():   
            # [- (M-1) / 2, ..., 0, ..., (M-1)/2]
            lp_coef = torch.arange(-self.half_k, self.half_k + 1, 
                                   device=cut_f.device)
            # [[[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ],
            #  [[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ]]
            lp_coef = lp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)
            
            hp_coef = torch.arange(-self.half_k, self.half_k + 1, 
                                   device=cut_f.device)
            hp_coef = hp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)
            
            # temporary buffer of [-1^n] for gain norm in hp_coef
            tmp_one = torch.pow(-1, hp_coef)
            
        # unnormalized filter coefs with hamming window
        lp_coef = cut_f * self.sinc(cut_f * lp_coef) * self.hamming_w(lp_coef)
        
        hp_coef = (self.sinc(hp_coef) \
                   - cut_f * self.sinc(cut_f * hp_coef)) \
                  * self.hamming_w(hp_coef)
        
        # normalize the coef to make gain at 0/pi is 0 dB
        # sum_n lp_coef[n]
        lp_coef_norm = torch.sum(lp_coef, axis=2).unsqueeze(-1)
        # sum_n hp_coef[n] * -1^n
        hp_coef_norm = torch.sum(hp_coef * tmp_one, axis=2).unsqueeze(-1)
        
        lp_coef = lp_coef / lp_coef_norm
        hp_coef = hp_coef / hp_coef_norm
        
        # return normed coef
        return lp_coef, hp_coef

class NeuralFilterBlock(nn.Module):
    """ Wrapper over a single filter block
    """
    def __init__(self, signal_size, hidden_size, kernel_size=3, conv_num=10):
        super(NeuralFilterBlock, self).__init__()
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv_num = conv_num
        self.dilation_s = [np.power(2, x) for x in np.arange(conv_num)]

        # ff layer to expand dimension
        self.l_ff_1 = nn.Linear(signal_size, hidden_size, bias=False)
        self.l_ff_1_tanh = nn.Tanh()
        
        # dilated conv layers
        tmp = [Conv1dKeepLength(hidden_size, hidden_size, x, kernel_size, causal=True, bias=False) \
               for x in self.dilation_s]
        self.l_convs = nn.ModuleList(tmp)
                
        # ff layer to de-expand dimension
        self.l_ff_2 = nn.Linear(hidden_size, hidden_size//4, bias=False)
        self.l_ff_2_tanh = nn.Tanh()
        self.l_ff_3 = nn.Linear(hidden_size//4, signal_size, bias=False)
        self.l_ff_3_tanh = nn.Tanh()        

        # a simple scale
        self.scale = nn.Parameter(torch.tensor([1/len(self.l_convs)]), requires_grad=False)
        return

    def forward(self, signal, context):
        """
        Assume: signal (batchsize=1, length, signal_size)
                context (batchsize=1, length, hidden_size)
        Output: (batchsize=1, length, signal_size)
        """
        # expand dimension
        tmp_hidden = self.l_ff_1_tanh(self.l_ff_1(signal))
        
        # loop over dilated convs
        # output of a d-conv is input + context + d-conv(input)
        for l_conv in self.l_convs:
            tmp_hidden = tmp_hidden + l_conv(tmp_hidden) + context
            
        # to be consistent with legacy configuration in CURRENNT
        tmp_hidden = tmp_hidden * self.scale
        
        # compress the dimesion and skip-add
        tmp_hidden = self.l_ff_2_tanh(self.l_ff_2(tmp_hidden))
        tmp_hidden = self.l_ff_3_tanh(self.l_ff_3(tmp_hidden))
        output_signal = tmp_hidden + signal
        
        return output_signal
    

class FilterModuleHnSincNSF(nn.Module):
    """ Filter for Hn-sinc-NSF
    FilterModuleHnSincNSF(signal_size, hidden_size, sinc_order = 31,
                          block_num = 5, kernel_size = 3, 
                          conv_num_in_block = 10)
    signal_size: signal dimension (should be 1)
    hidden_size: dimension of hidden features inside neural filter block
    sinc_order: order of the sinc filter
    block_num: number of neural filter blocks in harmonic branch
    kernel_size: kernel size in dilated CNN
    conv_num_in_block: number of d-conv1d in one neural filter block

    Usage:
    output = FilterModuleHnSincNSF(har_source, noi_source, cut_f, context)
    har_source: source for harmonic branch (batchsize, length, dim=1)
    noi_source: source for noise branch (batchsize, length, dim=1)
    cut_f: cut-off-frequency of sinc filters (batchsize, length, dim=1)
    context: hidden features to be added (batchsize, length, dim)
    output: (batchsize, length, dim=1)    
    """
    def __init__(self, signal_size, hidden_size, sinc_order = 31, \
                 block_num = 5, kernel_size = 3, conv_num_in_block = 10):
        super(FilterModuleHnSincNSF, self).__init__()        
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block
        self.sinc_order = sinc_order
        
        # filter blocks for harmonic branch
        tmp = [NeuralFilterBlock(signal_size, hidden_size, \
                                 kernel_size, conv_num_in_block) \
               for x in range(self.block_num)]
        self.l_har_blocks = nn.ModuleList(tmp)

        # filter blocks for noise branch (only one block, 5 sub-blocks)
        tmp = [NeuralFilterBlock(signal_size, hidden_size, \
                                 kernel_size, conv_num_in_block // 2) \
               for x in range(1)]
        self.l_noi_blocks = nn.ModuleList(tmp)

        # sinc filter generators and time-variant filtering layer
        self.l_sinc_coef = SincFilter(self.sinc_order)
        self.l_tv_filtering = TimeVarFIRFilter()

    def forward(self, har_component, noi_component, cond_feat, cut_f):
        # harmonic component
        for l_har_block in self.l_har_blocks:
            har_component = l_har_block(har_component, cond_feat)
        # noise componebt
        for l_noi_block in self.l_noi_blocks:
            noi_component = l_noi_block(noi_component, cond_feat)
        
        # get sinc filter coefficients
        lp_coef, hp_coef = self.l_sinc_coef(cut_f)

        # time-variant filtering
        har_signal = self.l_tv_filtering(har_component, lp_coef)
        noi_signal = self.l_tv_filtering(noi_component, hp_coef)

        # get output 
        return har_signal + noi_signal
        