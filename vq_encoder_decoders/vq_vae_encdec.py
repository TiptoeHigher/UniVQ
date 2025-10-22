"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vq_utils import timefreq_to_time, time_to_timefreq, SnakeActivation


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, frequency_indepence:bool, mid_channels=None, dropout:float=0.):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels
        
        kernel_size = (1, 3) if frequency_indepence else (3, 3)
        padding = (0, 1) if frequency_indepence else (1, 1)

        layers = [
            SnakeActivation(in_channels, 1), #SnakyGELU(in_channels, 2), #SnakeActivation(in_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Conv1d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            SnakeActivation(out_channels, 1), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Conv1d(mid_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout)
        ]
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x) + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frequency_indepence:bool,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 2), padding=padding,
            #           padding_mode='replicate'),
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                      padding_mode='replicate'),
            # nn.BatchNorm2d(out_channels),
            nn.BatchNorm1d(out_channels),
            SnakeActivation(out_channels, 1), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEDecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 frequency_indepence:bool,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            SnakeActivation(out_channels, 1), #SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 init_dim:int,
                 hid_dim: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 n_fft:int,
                 seq_len:int,
                 frequency_indepence:bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param bn: use of BatchNorm
        :param kwargs:
        """
        super().__init__()
        # self.pad_func = pad_func
        self.n_fft = n_fft
        self.hid_dim = hid_dim
        self.num_channels = num_channels
        self.seq_len = seq_len

        d = init_dim
        enc_layers = [VQVAEEncBlock(num_channels, d, frequency_indepence),]
        d *= 2
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            enc_layers.append(VQVAEEncBlock(d//2, d, frequency_indepence))
            for _ in range(n_resnet_blocks):
                enc_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d *= 2
        enc_layers.append(ResBlock(d//2, hid_dim, frequency_indepence, dropout=dropout))
        self.encoder = nn.Sequential(*enc_layers)

        # self.projection = nn.Linear(self.seq_len, self.seq_len*self.projection_rate)
        # self.incorporate_projection = nn.Conv2d(in_channels=self.num_channels,
        #                                         out_channels=self.num_channels*channel_upsample_rate,
        #                                         kernel_size=(1,1), padding=(0,0))
        # self.de_projection = nn.Linear(self.seq_len*self.projection_rate, self.seq_len)

        # self.is_num_tokens_updated = False
        # self.register_buffer('num_tokens', torch.tensor(0))
        # self.register_buffer('H_prime', torch.tensor(0))
        # self.register_buffer('W_prime', torch.tensor(0))
    
    def forward(self, x):
        """
        :param x: (b c l)
        """
        # in_channels = x.shape[1]
        # y = time_to_timefreq(x, self.n_fft, in_channels)# (b c h w)
        # start projection
        # x = self.projection(x)      # (b c h w)
        # x = x.reshape(x.shape[0], self.num_channels, self.projection_rate, -1)  # (b c h w)
        # x = self.incorporate_projection(x)  # (b c h w)

        # x = self.pad_func(x, copy=True)   # (b c h w)

        out = self.encoder(x)  # (b c l)
        # if not self.is_num_tokens_updated:
        #     self.H_prime = torch.tensor(out.shape[2])
        #     self.W_prime = torch.tensor(out.shape[3])
        #     self.num_tokens = self.H_prime * self.W_prime
        #     self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 init_dim:int,
                 hid_dim: int,
                 in_channel: int,
                 latent_len: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 n_fft:int,
                 seq_len: int,
                 frequency_indepence:bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        # self.pad_func = pad_func
        self.n_fft = n_fft
        self.hid_dim = hid_dim
        self.seq_len = seq_len
        self.latent_len = latent_len
        # self.up_sample_rate = 2**(int(round(np.log2(downsample_rate))) - 1 + 2)
        self.up_sample_rate = 2 ** (int(round(np.log2(downsample_rate))) - 1 + 1)

        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)
        
        d = int(init_dim * 2**(int(round(np.log2(downsample_rate))) - 1))  # enc_out_dim == dec_in_dim
        if round(np.log2(downsample_rate)) == 0:
            d = int(init_dim * 2**(int(round(np.log2(downsample_rate)))))
        # dec_layers = [ResBlock(hid_dim, d, frequency_indepence, dropout=dropout)]
        dec_layers = [ResBlock(self.hid_dim, d, frequency_indepence, dropout=dropout)]
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            for _ in range(n_resnet_blocks):
                dec_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d //= 2
            dec_layers.append(VQVAEDecBlock(2*d, d, frequency_indepence))
        dec_layers.append(nn.ConvTranspose1d(d, in_channel, kernel_size=4, stride=2, padding=1))
        # dec_layers.append(nn.ConvTranspose1d(d, hid_dim, kernel_size=4, stride=2, padding=1))
        # dec_layers.append(nn.ConvTranspose1d(hid_dim, hid_dim, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

        # # projection
        # self.de_projection_len_12 = nn.Linear(self.latent_len * self.up_sample_rate, 12)
        # self.de_projection_len_96 = nn.Linear(self.latent_len * self.up_sample_rate, 96)
        # self.de_projection_len_192 = nn.Linear(self.latent_len * self.up_sample_rate, 192)
        # self.de_projection_len_336 = nn.Linear(self.latent_len * self.up_sample_rate, 336)
        # self.de_projection_len_720 = nn.Linear(self.latent_len * self.up_sample_rate, 720)
        #
        # self.de_projection_dim = nn.Linear(hid_dim, in_channel)

        # self.linear_96 = nn.Linear(96, 96)
        # self.linear_192 = nn.Linear(192, 192)
        # self.linear_336 = nn.Linear(336, 336)
        # self.linear_720 = nn.Linear(720, 720)


    def forward(self, x):
        # x = x.reshape(x.shape[0], self.hid_dim, self.projection_rate, self.latent_len)

        out_96 = self.decoder(x[:, :, :12])  # down_sample_rate = 8, 96=8*12
        out_192 = self.decoder(x[:, :, :24])  # down_sample_rate = 8, 192=8*24
        out_336 = self.decoder(x[:, :, :42])  # down_sample_rate = 8, 336=8*42
        out_720 = self.decoder(x[:, :, :90])  # down_sample_rate = 8, 720=8*90

        # out = out.reshape(x.shape[0], self.hid_dim, self.projection_rate*self.latent_len* self.up_sample_rate)
        #
        # # dim projection
        # out = out.permute(0, 2, 1)  # (b l c)
        # out = self.de_projection_dim(out)  # (b l c)
        # out = out.permute(0, 2, 1)  # (b 1 l)

        # out_96 = self.de_projection_len_96(out)
        # out_192 = self.de_projection_len_192(out)
        # out_336 = self.de_projection_len_336(out)
        # out_720 = self.de_projection_len_720(out)

        # out_96 = self.linear_96(out_96)
        # out_192 = self.linear_192(out_192)
        # out_336 = self.linear_336(out_336)
        # out_720 = self.linear_720(out_720)

        return out_96, out_192, out_336, out_720
