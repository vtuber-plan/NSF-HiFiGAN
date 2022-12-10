

from functools import reduce
import operator
from typing import List, Union
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d

from nsf_hifigan.model.generators.cond_module import CondModuleHnSincNSF
from nsf_hifigan.model.generators.filter_module import FilterModuleHnSincNSF
from nsf_hifigan.model.generators.source_module import SourceModuleHnNSF

class NSFHiFiGANGenerator(torch.nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim: int, out_dim: int,
            upsampling_rate: int,
            sampling_rate: int,
            sine_amp: float=0.1,
            noise_std: float=0.003,
            hidden_dim: int=64,
            cnn_kernel_s: int=3,
            filter_block_num: int=5,
            cnn_num_in_block: int=10,
            harmonic_num: int=7,
            sinc_order: int=31
        ):
        super(NSFHiFiGANGenerator, self).__init__()

        self.input_dim = in_dim
        self.output_dim = out_dim

        # configurations
        # amplitude of sine waveform (for each harmonic)
        self.sine_amp = sine_amp
        # standard deviation of Gaussian noise for additive noise
        self.noise_std = noise_std
        # dimension of hidden features in filter blocks
        self.hidden_dim = hidden_dim
        # upsampling rate on input acoustic features (16kHz * 5ms = 80)
        # assume input_reso has the same value
        self.upsampling_rate = upsampling_rate
        # sampling rate (Hz)
        self.sampling_rate = sampling_rate
        # CNN kernel size in filter blocks        
        self.cnn_kernel_s = cnn_kernel_s
        # number of filter blocks (for harmonic branch)
        # noise branch only uses 1 block
        self.filter_block_num = filter_block_num
        # number of dilated CNN in each filter block
        self.cnn_num_in_block = cnn_num_in_block
        # number of harmonic overtones in source
        self.harmonic_num = harmonic_num
        # order of sinc-windowed-FIR-filter
        self.sinc_order = sinc_order

        # the three modules
        self.m_cond = CondModuleHnSincNSF(self.input_dim, self.hidden_dim, self.upsampling_rate, cnn_kernel_s=self.cnn_kernel_s)

        self.m_source = SourceModuleHnNSF(self.sampling_rate, self.harmonic_num, self.sine_amp, self.noise_std)
        
        self.m_filter = FilterModuleHnSincNSF(self.output_dim, self.hidden_dim, self.sinc_order, self.filter_block_num, \
                                            self.cnn_kernel_s, self.cnn_num_in_block)
        # loss function on spectra
        # self.m_aux_loss = LossAuxGen()
    
    def forward(self, feat, f0):
        """ definition of forward method 
        Assume x (batchsize=1, length, dim)
        Return output(batchsize=1, length)
        """
        # condition module
        # feature-to-filter-block, f0-up-sampled, cut-off-f-for-sinc,
        # hidden-feature-for-cut-off-f
        cond_feat, f0_upsamped, cut_f, hid_cut_f = self.m_cond(feat, f0)

        # source module
        # harmonic-source, noise-source (for noise branch), uv
        har_source, noi_source, uv = self.m_source(f0_upsamped)
        
        # neural filter module (including sinc-based FIR filtering)
        # output
        output = self.m_filter(har_source, noi_source, cond_feat, cut_f)
        
        return output
    
    def loss_aux(self, nat_wav, gen_tuple, data_in):
        return self.m_aux_loss.compute(gen_tuple, nat_wav)

