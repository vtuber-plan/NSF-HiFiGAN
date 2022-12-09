

from functools import reduce
import operator
from typing import List, Union
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d

class NSFHiFiGANGenerator(torch.nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(NSFHiFiGANGenerator, self).__init__()

        self.input_dim = in_dim
        self.output_dim = out_dim

        # configurations
        # amplitude of sine waveform (for each harmonic)
        self.sine_amp = 0.1
        # standard deviation of Gaussian noise for additive noise
        self.noise_std = 0.003
        # dimension of hidden features in filter blocks
        self.hidden_dim = 64
        # upsampling rate on input acoustic features (16kHz * 5ms = 80)
        # assume input_reso has the same value
        self.upsamp_rate = prj_conf.input_reso[0]
        # sampling rate (Hz)
        self.sampling_rate = prj_conf.wav_samp_rate
        # CNN kernel size in filter blocks        
        self.cnn_kernel_s = 3
        # number of filter blocks (for harmonic branch)
        # noise branch only uses 1 block
        self.filter_block_num = 5
        # number of dilated CNN in each filter block
        self.cnn_num_in_block = 10
        # number of harmonic overtones in source
        self.harmonic_num = 7
        # order of sinc-windowed-FIR-filter
        self.sinc_order = 31

        # the three modules
        self.m_cond = CondModuleHnSincNSF(self.input_dim, \
                                          self.hidden_dim, \
                                          self.upsamp_rate, \
                                          cnn_kernel_s=self.cnn_kernel_s)

        self.m_source = SourceModuleHnNSF(self.sampling_rate, 
                                          self.harmonic_num, 
                                          self.sine_amp, self.noise_std)
        
        self.m_filter = FilterModuleHnSincNSF(self.output_dim, \
                                              self.hidden_dim, \
                                              self.sinc_order, \
                                              self.filter_block_num, \
                                              self.cnn_kernel_s, \
                                              self.cnn_num_in_block)
        # loss function on spectra
        self.m_aux_loss = LossAuxGen()

        # done
        return
    
    def forward(self, x):
        """ definition of forward method 
        Assume x (batchsize=1, length, dim)
        Return output(batchsize=1, length)
        """
        # assume x[:, :, -1] is F0, denormalize F0
        f0 = x[:, :, -1:]
        # normalize the input features data
        feat = x

        # condition module
        # feature-to-filter-block, f0-up-sampled, cut-off-f-for-sinc,
        #  hidden-feature-for-cut-off-f
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

