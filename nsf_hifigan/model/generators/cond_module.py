
import torch
from torch import nn
from torch.nn import functional as F

from nsf_hifigan.model.modules.conv1_keep_length import Conv1dKeepLength

from ..modules.moving_average import MovingAverage

class UpSampleLayer(nn.Module):
    """ Wrapper over up-sampling
    Input tensor: (batchsize=1, length, dim)
    Ouput tensor: (batchsize=1, length * up-sampling_factor, dim)
    """
    def __init__(self, feature_dim, up_sampling_factor, smoothing=False):
        super(UpSampleLayer, self).__init__()
        # wrap a up_sampling layer
        self.scale_factor = up_sampling_factor
        self.l_upsamp = nn.Upsample(scale_factor=self.scale_factor)
        if smoothing:
            self.l_ave1 = MovingAverage(feature_dim, self.scale_factor)
            self.l_ave2 = MovingAverage(feature_dim, self.scale_factor)
        else:
            self.l_ave1 = nn.Identity()
            self.l_ave2 = nn.Identity()
        return
    
    def forward(self, x):
        # permute to (batchsize=1, dim, length)
        up_sampled_data = self.l_upsamp(x.permute(0, 2, 1))

        # permute it backt to (batchsize=1, length, dim)
        # and do two moving average
        return self.l_ave1(self.l_ave2(up_sampled_data.permute(0, 2, 1)))
    

class CondModuleHnSincNSF(nn.Module):
    """ Condition module for hn-sinc-NSF

    Upsample and transform input features
    CondModuleHnSincNSF(input_dimension, output_dimension, up_sample_rate,
               blstm_dimension = 64, cnn_kernel_size = 3)
    
    Spec, F0, cut_off_freq = CondModuleHnSincNSF(features, F0)

    Both input features should be frame-level features
    If x doesn't contain F0, just ignore the returned F0
    
    CondModuleHnSincNSF(input_dim, output_dim, up_sample, 
                        blstm_s = 64, cnn_kernel_s = 3, 
                        voiced_threshold = 0):

    input_dim: sum of dimensions of input features
    output_dim: dim of the feature Spec to be used by neural filter-block
    up_sample: up sampling rate of input features
    blstm_s: dimension of the features from blstm (default 64)
    cnn_kernel_s: kernel size of CNN in condition module (default 3)
    voiced_threshold: f0 > voiced_threshold is voiced, otherwise unvoiced
    """
    def __init__(self, input_dim, output_dim, up_sample, \
                 blstm_s = 64, cnn_kernel_s = 3, voiced_threshold = 0):
        super(CondModuleHnSincNSF, self).__init__()

        # input feature dimension
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_sample = up_sample
        self.blstm_s = blstm_s
        self.cnn_kernel_s = cnn_kernel_s
        self.cut_f_smooth = up_sample * 4
        self.voiced_threshold = voiced_threshold

        # the blstm layer
        self.l_blstm = nn.LSTM(input_dim, self.blstm_s // 2, bidirectional=True, batch_first=True)

        # the CNN layer (+1 dim for cut_off_frequence of sinc filter)
        self.l_conv1d = Conv1dKeepLength(self.blstm_s, \
                                         self.output_dim, \
                                         dilation_s = 1, \
                                         kernel_s = self.cnn_kernel_s)
        # Upsampling layer for hidden features
        self.l_upsamp = UpSampleLayer(self.output_dim, \
                                      self.up_sample, True)
        # separate layer for up-sampling normalized F0 values
        self.l_upsamp_f0_hi = UpSampleLayer(1, self.up_sample, True)
        
        # Upsampling for F0: don't smooth up-sampled F0
        self.l_upsamp_F0 = UpSampleLayer(1, self.up_sample, False)

        # Another smoothing layer to smooth the cut-off frequency
        # for sinc filters. Use a larger window to smooth
        self.l_cut_f_smooth = MovingAverage(1, self.cut_f_smooth)

    def get_cut_f(self, hidden_feat, f0):
        """ cut_f = get_cut_f(self, feature, f0)
        feature: (batchsize, length, dim=1)
        f0: (batchsize, length, dim=1)        
        """ 
        # generate uv signal
        uv = torch.ones_like(f0) * (f0 > self.voiced_threshold)
        # hidden_feat is between (-1, 1) after conv1d with tanh
        # (-0.2, 0.2) + 0.3 = (0.1, 0.5)
        # voiced:   (0.1, 0.5) + 0.4 = (0.5, 0.9)
        # unvoiced: (0.1, 0.5) = (0.1, 0.5)
        return hidden_feat * 0.2 + uv * 0.4 + 0.3
        
    
    def forward(self, feature, f0):
        """ spec, f0 = forward(self, feature, f0)
        feature: (batchsize, length, dim)
        f0: (batchsize, length, dim=1), which should be F0 at frame-level
        
        spec: (batchsize, length, self.output_dim), at wave-level
        f0: (batchsize, length, 1), at wave-level
        """ 
        feature_h, feature_c = self.l_blstm(feature)
        tmp = self.l_upsamp(self.l_conv1d(feature_h))
        
        # concatenat normed F0 with hidden spectral features
        context = torch.cat((tmp[:, :, 0:self.output_dim-1], \
                             self.l_upsamp_f0_hi(feature[:, :, -1:])), \
                            dim=2)
        
        # hidden feature for cut-off frequency
        hidden_cut_f = tmp[:, :, self.output_dim-1:]

        # directly up-sample F0 without smoothing
        f0_upsamp = self.l_upsamp_F0(f0)

        # get the cut-off-frequency from output of CNN
        cut_f = self.get_cut_f(hidden_cut_f, f0_upsamp)
        # smooth the cut-off-frequency using fixed average smoothing
        cut_f_smoothed = self.l_cut_f_smooth(cut_f)

        # return
        return context, f0_upsamp, cut_f_smoothed, hidden_cut_f
