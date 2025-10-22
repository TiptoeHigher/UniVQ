import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.onnx.symbolic_opset11 import unsqueeze
from torch.cuda.amp import autocast
from vector_quantize_pytorch import FSQ

from models import TimesNet, Autoformer, Transformer, Nonstationary_Transformer, DLinear, FEDformer, Informer, LightTS, Reformer, ETSformer, \
PatchTST, Pyraformer, MICN, Crossformer, FiLM, iTransformer, Koopa, TiDE, FreTS, MambaSimple, TimeMixer, TSMixer, SegRNN, TemporalFusionTransformer, \
SCINet
# import wandb
# import pytorch_lightning as pl

from vq_encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantization import VectorQuantize
from vq_utils import compute_downsample_rate, timefreq_to_time, time_to_timefreq, \
    quantize, linear_warmup_cosine_annealingLR, load_yaml_param_settings


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        """
        :param input_length: length of input time series
        :param config: configs/config.yaml
        :param n_train_samples: number of training samples
        """

        self.in_channels = 1
        self.input_length = args.seq_len
        self.pred_length = args.pred_len
        # self.projection_rate = args.projection_rate
        self.level = args.level
        # self.channel_upsample_rate = args.channel_upsample_rate
        self.enc_in = args.enc_in
        self.config = load_yaml_param_settings(args.vq_configs)

        self.n_fft = self.config['VQ-VAE']['n_fft']
        init_dim = self.config['encoder']['init_dim']
        # hid_dim = self.config['encoder']['hid_dim']
        hid_dim = args.code_dim
        print('discrete dim:', hid_dim)
        # downsampled_width_l = self.config['encoder']['downsampled_width']['lf']
        # downsampled_width_h = self.config['encoder']['downsampled_width']['hf']
        # downsample_rate_l = compute_downsample_rate(self.input_length, self.n_fft, downsampled_width_l)
        # downsample_rate_h = compute_downsample_rate(self.input_length, self.n_fft, downsampled_width_h)
        downsample_rate_l = args.downsample_rate

        self.latent_len = self.input_length
        for i in range(int(round(np.log2(downsample_rate_l)))):
            self.latent_len = int(self.latent_len // 2)
        self.out_latent_len = args.out_latent_len

        self.predictor_dict = {
            'PatchTST': PatchTST
        }
        # choose the predictor
        self.predictor = self.predictor_dict[args.predictor].Model(args, seq_len=self.latent_len,
                                                                   pred_len=self.out_latent_len).float()

        # encoder
        self.encoder = VQVAEEncoder(init_dim, hid_dim, self.in_channels, downsample_rate_l,
                                      self.config['encoder']['n_resnet_blocks'], self.n_fft,
                                    self.input_length, frequency_indepence=False)
        # self.encoder_l = VQVAEEncoder(init_dim, hid_dim, 2 * self.in_channels, downsample_rate_l,
        #                               self.config['encoder']['n_resnet_blocks'], zero_pad_high_freq, self.n_fft,
        #                               frequency_indepence=False)
        # self.encoder_h = VQVAEEncoder(init_dim, hid_dim, 2 * self.in_channels, downsample_rate_h,
        #                               self.config['encoder']['n_resnet_blocks'], zero_pad_low_freq, self.n_fft,
        #                               frequency_indepence=False)

        # quantizer
        # self.vq_model = VectorQuantize(hid_dim, self.config['VQ-VAE']['codebook_sizes']['lf'], **self.config['VQ-VAE'])
        # self.vq_model_l = VectorQuantize(hid_dim, self.config['VQ-VAE']['codebook_sizes']['lf'], **self.config['VQ-VAE'])
        # self.vq_model_h = VectorQuantize(hid_dim, self.config['VQ-VAE']['codebook_sizes']['hf'], **self.config['VQ-VAE'])
        self.quantizer = FSQ(levels=[self.level for _ in range(hid_dim)])

        # projection
        # self.de_projection = nn.Linear(hid_dim * self.q_H * self.q_W, self.input_length*self.enc_in)
        self.projection = nn.Linear(hid_dim, self.enc_in)
        self.de_projection = nn.Linear(self.enc_in, hid_dim)
        # self.projection = VQVAEEncoder(init_dim, hid_dim, self.in_channels, downsample_rate_l,
        #                             self.config['encoder']['n_resnet_blocks'], zero_pad_high_freq, self.n_fft,
        #                             self.input_length, self.projection_rate, self.channel_upsample_rate,
        #                             frequency_indepence=False)
        # self.de_projection = VQVAEDecoder(init_dim, hid_dim, self.in_channels, self.channel_upsample_rate, downsample_rate_l,
        #                               self.config['decoder']['n_resnet_blocks'], self.input_length, zero_pad_high_freq,
        #                               self.n_fft, self.in_channels, frequency_indepence=True)

        # decoder
        self.decoder = VQVAEDecoder(init_dim, hid_dim, self.in_channels, self.latent_len, downsample_rate_l,
                                          self.config['decoder']['n_resnet_blocks'],
                                          self.n_fft, self.pred_length, frequency_indepence=True)
        # self.decoder_l = VQVAEDecoder(init_dim, hid_dim, 2 * self.in_channels, downsample_rate_l,
        #                               self.config['decoder']['n_resnet_blocks'], self.input_length, zero_pad_high_freq,
        #                               self.n_fft, self.in_channels, frequency_indepence=True)
        # self.decoder_h = VQVAEDecoder(init_dim, hid_dim, 2 * self.in_channels, downsample_rate_h,
        #                               self.config['decoder']['n_resnet_blocks'], self.input_length, zero_pad_low_freq, self.n_fft,
        #                               self.in_channels, frequency_indepence=True)

        # sigmoid
        # self.sigma = nn.Sigmoid()


    def forward(self, batch, batch_idx, return_x_rec: bool = False):
        """
        :param x: input time series (b c l)
        """
        # x, y = batch

        x = batch
        x = x.permute(0, 2, 1)

        # recons_loss = {'LF.time': 0., 'HF.time': 0.}
        # vq_losses = {'LF': None, 'HF': None}
        # perplexities = {'LF': 0., 'HF': 0.}

        # # STFT
        # in_channels = x.shape[1]
        # xf = time_to_timefreq(x, self.n_fft, in_channels)  # (b c h w)
        # u_l = zero_pad_high_freq(xf)  # (b c h w)
        # x_l = F.interpolate(timefreq_to_time(u_l, self.n_fft, in_channels), self.input_length, mode='linear')  # (b c l)
        # u_h = zero_pad_low_freq(xf)  # (b c h w)
        # x_h = F.interpolate(timefreq_to_time(u_h, self.n_fft, in_channels), self.input_length, mode='linear')  # (b c l)
        #
        # # LF
        # z_l = self.encoder_l(x)
        # z_q_l, s_l, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
        # xhat_l = self.decoder_l(z_q_l)  # (b c l)
        #
        # # HF
        # z_h = self.encoder_h(x)
        # z_q_h, s_h, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
        # xhat_h = self.decoder_h(z_q_h)  # (b c l)

        # encode
        z = self.encoder(x)

        # quantize
        # z_q, s, vq_loss, perplexity = quantize(z, self.vq_model)

        # quantize by myself
        # z_q = (self.level-1)*self.sigma(z) + \
        #       (torch.round((self.level-1)*self.sigma(z))-(self.level-1)*self.sigma(z)).detach()

        # quantize by vector_quantize_pytorch
        z = z.permute(0, 2, 1)
        # z = z.reshape(z.shape[0], -1, z.shape[-1])
        z_q, embed_ind = self.quantizer(z)

        # projection and reshape
        # z_q = z_q.reshape(z_q.shape[0], -1)
        # z_q = self.de_projection(z_q)
        # z_q = z_q.reshape(z_q.shape[0], self.input_length, self.enc_in)
        # # z_q = z_q.reshape(z_q.shape[0], -1)
        # # z_q = z_q.unsqueeze(-1)

        # projection and reshape -lite
        z_q = self.projection(z_q)

        # predict and quantize
        z_q_prd = self.predictor(z_q)
        z_q_prd = self.de_projection(z_q_prd)
        z_q_prd, prd_embed_ind = self.quantizer(z_q_prd)
        z_q_prd = z_q_prd.permute(0, 2, 1)

        # decode
        xhat_96, xhat_192, xhat_336, xhat_720 = self.decoder(z_q_prd)  # (b c l)

        xhat_96 = xhat_96.permute(0, 2, 1)
        xhat_192 = xhat_192.permute(0, 2, 1)
        xhat_336 = xhat_336.permute(0, 2, 1)
        xhat_720 = xhat_720.permute(0, 2, 1)


        # if return_x_rec:
        #     x_rec = xhat_l + xhat_h  # (b c l)
        #     return x_rec  # (b c l)
        #
        # recons_loss['LF.time'] = F.mse_loss(x_l, xhat_l)
        # perplexities['LF'] = perplexity_l
        # vq_losses['LF'] = vq_loss_l
        #
        # recons_loss['HF.time'] = F.l1_loss(x_h, xhat_h)
        # perplexities['HF'] = perplexity_h
        # vq_losses['HF'] = vq_loss_h

        # # plot `x` and `xhat`
        # if not self.training and batch_idx == 0:
        #     b = np.random.randint(0, x_h.shape[0])
        #     c = np.random.randint(0, x_h.shape[1])
        #
        #     alpha = 0.7
        #     n_rows = 3
        #     fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2 * n_rows))
        #     plt.suptitle(f'step-{self.global_step} | channel idx:{c} \n (blue:GT, orange:reconstructed)')
        #     axes[0].plot(x_l[b, c].cpu(), alpha=alpha)
        #     axes[0].plot(xhat_l[b, c].detach().cpu(), alpha=alpha)
        #     axes[0].set_title(r'$x_l$ (LF)')
        #     axes[0].set_ylim(-4, 4)
        #
        #     axes[1].plot(x_h[b, c].cpu(), alpha=alpha)
        #     axes[1].plot(xhat_h[b, c].detach().cpu(), alpha=alpha)
        #     axes[1].set_title(r'$x_h$ (HF)')
        #     axes[1].set_ylim(-4, 4)
        #
        #     axes[2].plot(x_l[b, c].cpu() + x_h[b, c].cpu(), alpha=alpha)
        #     axes[2].plot(xhat_l[b, c].detach().cpu() + xhat_h[b, c].detach().cpu(), alpha=alpha)
        #     axes[2].set_title(r'$x$ (LF+HF)')
        #     axes[2].set_ylim(-4, 4)
        #
        #     plt.tight_layout()
        #     # wandb.log({"x vs x_rec (val)": wandb.Image(plt)})
        #     plt.close()

        # return recons_loss, vq_losses, perplexities
        # print('using VQ! current vq_loss: {}'.format(_vq_loss))
        return xhat_96, xhat_192, xhat_336, xhat_720

    # def training_step(self, batch, batch_idx):
    #     recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
    #     loss = (recons_loss['LF.time'] + recons_loss['HF.time']) + vq_losses['LF']['loss'] + vq_losses['HF']['loss']
    #
    #     # lr scheduler
    #     sch = self.lr_schedulers()
    #     sch.step()
    #
    #     # log
    #     loss_hist = {'loss': loss,
    #                  'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
    #                  'recons_loss.LF.time': recons_loss['LF.time'],
    #                  'recons_loss.HF.time': recons_loss['HF.time'],
    #
    #                  'commit_loss.LF': vq_losses['LF']['commit_loss'],
    #                  'commit_loss.HF': vq_losses['HF']['commit_loss'],
    #                  'perplexity.LF': perplexities['LF'],
    #                  'perplexity.HF': perplexities['HF'],
    #                  }
    #
    #     # log
    #     for k in loss_hist.keys():
    #         self.log(f'train/{k}', loss_hist[k])
    #
    #     return loss_hist
    #
    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     self.eval()
    #
    #     recons_loss, vq_losses, perplexities = self.forward(batch, batch_idx)
    #     loss = (recons_loss['LF.time'] + recons_loss['HF.time']) + vq_losses['LF']['loss'] + vq_losses['HF']['loss']
    #
    #     # log
    #     loss_hist = {'loss': loss,
    #                  'recons_loss.time': recons_loss['LF.time'] + recons_loss['HF.time'],
    #                  'recons_loss.LF.time': recons_loss['LF.time'],
    #                  'recons_loss.HF.time': recons_loss['HF.time'],
    #
    #                  'commit_loss.LF': vq_losses['LF']['commit_loss'],
    #                  'commit_loss.HF': vq_losses['HF']['commit_loss'],
    #                  'perplexity.LF': perplexities['LF'],
    #                  'perplexity.HF': perplexities['HF'],
    #                  }
    #
    #     # log
    #     for k in loss_hist.keys():
    #         self.log(f'val/{k}', loss_hist[k])
    #
    #     return loss_hist
    #
    # def configure_optimizers(self):
    #     opt = torch.optim.AdamW(self.parameters(), lr=self.config['exp_params']['lr'])
    #     scheduler = linear_warmup_cosine_annealingLR(opt, self.config['trainer_params']['max_steps']['stage1'],
    #                                                  self.config['exp_params']['linear_warmup_rate'])
    #     return {'optimizer': opt, 'lr_scheduler': scheduler}


