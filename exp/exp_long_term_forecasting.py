from tensorboardX import SummaryWriter
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from thop import profile
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

            # 多卡测试
            if self.args.use_multi_gpu:
                self.model.load_state_dict(torch.load(self.args.checkpoint_path, map_location='cuda:0'))

            # 单卡测试
            else:
                # 多卡训练之后需要对checkpoint做一下重命名
                # 加载原始检查点
                checkpoint = torch.load(self.args.checkpoint_path, map_location='cuda:0')

                # 创建一个新的 state_dict，不带 'module.' 前缀
                new_state_dict = {}
                for key, value in checkpoint.items():
                    # 去掉 'module.' 前缀
                    new_key = key.replace('module.', '', 1)
                    new_state_dict[new_key] = value

                # 更新检查点中的 state_dict
                checkpoint = new_state_dict

                self.model.load_state_dict(checkpoint)

        preds_96 = []
        trues_96 = []
        mse_96 = []
        mae_96 = []

        preds_192 = []
        trues_192 = []
        mse_192 = []
        mae_192 = []

        preds_336 = []
        trues_336 = []
        mse_336 = []
        mae_336 = []

        preds_720 = []
        trues_720 = []
        mse_720 = []
        mae_720 = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y_96, batch_y_192, batch_y_336, batch_y_720) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y_96 = batch_y_96.float().to(self.device)
                batch_y_192 = batch_y_192.float().to(self.device)
                batch_y_336 = batch_y_336.float().to(self.device)
                batch_y_720 = batch_y_720.float().to(self.device)

                # count the channels
                num_channels = batch_x.shape[-1]

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_96 = []
                        outputs_192 = []
                        outputs_336 = []
                        outputs_720 = []
                        for t in range(num_channels):
                            out_96, out_192, out_336, out_720 = self.model(batch_x[:, :, t].unsqueeze(-1), i)
                            outputs_96.append(out_96)
                            outputs_192.append(out_192)
                            outputs_336.append(out_336)
                            outputs_720.append(out_720)

                        outputs_96 = torch.cat(outputs_96, dim=-1)
                        outputs_192 = torch.cat(outputs_192, dim=-1)
                        outputs_336 = torch.cat(outputs_336, dim=-1)
                        outputs_720 = torch.cat(outputs_720, dim=-1)
                else:
                    outputs_96 = []
                    outputs_192 = []
                    outputs_336 = []
                    outputs_720 = []
                    for t in range(num_channels):
                        out_96, out_192, out_336, out_720 = self.model(batch_x[:, :, t].unsqueeze(-1), i)
                        outputs_96.append(out_96)
                        outputs_192.append(out_192)
                        outputs_336.append(out_336)
                        outputs_720.append(out_720)

                    outputs_96 = torch.cat(outputs_96, dim=-1)
                    outputs_192 = torch.cat(outputs_192, dim=-1)
                    outputs_336 = torch.cat(outputs_336, dim=-1)
                    outputs_720 = torch.cat(outputs_720, dim=-1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_96 = outputs_96[:, -96:, f_dim:]
                outputs_192 = outputs_192[:, -192:, f_dim:]
                outputs_336 = outputs_336[:, -336:, f_dim:]
                outputs_720 = outputs_720[:, -720:, f_dim:]

                batch_y_96 = batch_y_96[:, -96:, f_dim:].to(self.device)
                zero_sample_mask_96 = torch.all(batch_y_96 == 0, dim=(1, 2))
                batch_y_96_mask = batch_y_96[~zero_sample_mask_96]
                outputs_96_mask = outputs_96[~zero_sample_mask_96]

                batch_y_192 = batch_y_192[:, -192:, f_dim:].to(self.device)
                zero_sample_mask_192 = torch.all(batch_y_192 == 0, dim=(1, 2))
                batch_y_192_mask = batch_y_192[~zero_sample_mask_192]
                outputs_192_mask = outputs_192[~zero_sample_mask_192]

                batch_y_336 = batch_y_336[:, -336:, f_dim:].to(self.device)
                zero_sample_mask_336 = torch.all(batch_y_336 == 0, dim=(1, 2))
                batch_y_336_mask = batch_y_336[~zero_sample_mask_336]
                outputs_336_mask = outputs_336[~zero_sample_mask_336]

                batch_y_720 = batch_y_720[:, -720:, f_dim:].to(self.device)
                zero_sample_mask_720 = torch.all(batch_y_720 == 0, dim=(1, 2))
                batch_y_720_mask = batch_y_720[~zero_sample_mask_720]
                outputs_720_mask = outputs_720[~zero_sample_mask_720]

                outputs_96_mask = outputs_96_mask.detach().cpu().numpy()
                outputs_192_mask = outputs_192_mask.detach().cpu().numpy()
                outputs_336_mask = outputs_336_mask.detach().cpu().numpy()
                outputs_720_mask = outputs_720_mask.detach().cpu().numpy()

                batch_y_96_mask = batch_y_96_mask.detach().cpu().numpy()
                batch_y_192_mask = batch_y_192_mask.detach().cpu().numpy()
                batch_y_336_mask = batch_y_336_mask.detach().cpu().numpy()
                batch_y_720_mask = batch_y_720_mask.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape_96_mask = outputs_96_mask.shape
                    outputs_96_mask = test_data.inverse_transform(outputs_96_mask.reshape(shape_96_mask[0] * shape_96_mask[1], -1)).reshape(shape_96_mask)
                    batch_y_96_mask = test_data.inverse_transform(batch_y_96_mask.reshape(shape_96_mask[0] * shape_96_mask[1], -1)).reshape(shape_96_mask)

                    shape_192_mask = outputs_192_mask.shape
                    outputs_192_mask = test_data.inverse_transform(outputs_192_mask.reshape(shape_192_mask[0] * shape_192_mask[1], -1)).reshape(shape_192_mask)
                    batch_y_192_mask = test_data.inverse_transform(batch_y_192_mask.reshape(shape_192_mask[0] * shape_192_mask[1], -1)).reshape(shape_192_mask)

                    shape_336_mask = outputs_336_mask.shape
                    outputs_336_mask = test_data.inverse_transform(outputs_336_mask.reshape(shape_336_mask[0] * shape_336_mask[1], -1)).reshape(shape_336_mask)
                    batch_y_336_mask = test_data.inverse_transform(batch_y_336_mask.reshape(shape_336_mask[0] * shape_336_mask[1], -1)).reshape(shape_336_mask)

                    shape_720_mask = outputs_720_mask.shape
                    outputs_720_mask = test_data.inverse_transform(outputs_720_mask.reshape(shape_720_mask[0] * shape_720_mask[1], -1)).reshape(shape_720_mask)
                    batch_y_720_mask = test_data.inverse_transform(batch_y_720_mask.reshape(shape_720_mask[0] * shape_720_mask[1], -1)).reshape(shape_720_mask)

                mae96, mse96 = metric(outputs_96_mask, batch_y_96_mask)
                mse_96.append(mse96)
                mae_96.append(mae96)

                mae192, mse192 = metric(outputs_192_mask, batch_y_192_mask)
                mse_192.append(mse192)
                mae_192.append(mae192)

                mae336, mse336 = metric(outputs_336_mask, batch_y_336_mask)
                mse_336.append(mse336)
                mae_336.append(mae336)

                mae720, mse720 = metric(outputs_720_mask, batch_y_720_mask)
                mse_720.append(mse720)
                mae_720.append(mae720)

        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds_96.shape[0]):
                x = preds_96[i].reshape(-1,1)
                y = trues_96[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'
            
        # save 96
        # mae_96, mse_96, rmse_96, mape_96, mspe_96, rse_96, corr_96 = metric(preds_96, trues_96)
        mae_96_array = np.array(mae_96)
        mse_96_array = np.array(mse_96)
        _mse_96 = np.nanmean(mse_96_array)
        _mae_96 = np.nanmean(mae_96_array)
        print('mse_96:{}, mae_96:{}, dtw_96:{}'.format(_mse_96, _mae_96, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse_96:{}, mae_96:{}'.format(_mse_96, _mae_96))
        f.write('\n')
        f.write('\n')
        f.close()

        # save 192
        # mae_192, mse_192, rmse_192, mape_192, mspe_192, rse_192, corr_192 = metric(preds_192, trues_192)
        mae_192_array = np.array(mae_192)
        mse_192_array = np.array(mse_192)
        _mse_192 = np.nanmean(mse_192_array)
        _mae_192 = np.nanmean(mae_192_array)
        print('mse_192:{}, mae_192:{}, dtw_192:{}'.format(_mse_192, _mae_192, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse_192:{}, mae_192:{}'.format(_mse_192, _mae_192))
        f.write('\n')
        f.write('\n')
        f.close()

        # save 336
        # mae_336, mse_336, rmse_336, mape_336, mspe_336, rse_336, corr_336 = metric(preds_336, trues_336)
        mae_336_array = np.array(mae_336)
        mse_336_array = np.array(mse_336)
        _mse_336 = np.nanmean(mse_336_array)
        _mae_336 = np.nanmean(mae_336_array)
        print('mse_336:{}, mae_336:{}, dtw_336:{}'.format(_mse_336, _mae_336, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse_336:{}, mae_336:{}'.format(_mse_336, _mae_336))
        f.write('\n')
        f.write('\n')
        f.close()


        # save 720
        # mae_720, mse_720, rmse_720, mape_720, mspe_720, rse_720, corr_720 = metric(preds_720, trues_720)
        mae_720_array = np.array(mae_720)
        mse_720_array = np.array(mse_720)
        _mse_720 = np.nanmean(mse_720_array)
        _mae_720 = np.nanmean(mae_720_array)
        print('mse_720:{}, mae_720:{}, dtw_720:{}'.format(_mse_720, _mae_720, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse_720:{}, mae_720:{}'.format(_mse_720, _mae_720))
        f.write('\n')
        f.write('\n')
        f.close()


        return

