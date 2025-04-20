import csv

import numpy as np
import pandas as pd
import torch

from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def compute_stats(generator_model, multimodal_dict, model, logger, cfg, save_results=False):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def get_prediction(data, model_select, mode='whole', slice_num=5, cfg=None):
        if cfg.dataset == 'assemble':
            traj_np = data.reshape([data.shape[0], cfg.t_total, -1])
            traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        else:
            traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
            traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
            traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)

        traj_est = torch.zeros(traj.shape, device=cfg.device, dtype=torch.float32)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]

        if mode == 'whole':
            mode_dict, traj_dct, traj_dct_cond, vel_acc_pad = sample_preprocessing(traj, cfg, mode='metrics')
            if cfg.generator == 'diffusion':
                sampled_motion = generator_model.sample_ddim(model_select,
                                                       traj_dct,
                                                       traj_dct_cond,
                                                       mode_dict,
                                                       vel_acc_pad=vel_acc_pad)
            elif cfg.generator == 'flow_matching':
                generator_model.update_sampling_model(model)
                sampled_motion = generator_model.sample_fm(mode_dict, traj_dct, traj_dct_cond)

            if cfg.use_dct:
                traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            else:
                traj_est = sampled_motion
        else:
            n = traj.shape[0] // slice_num
            for s in range(slice_num):
                if s != slice_num - 1:
                    traj_tmp = traj[s * n:(s + 1) * n, :, :]
                else:
                    traj_tmp = traj[s * n:, :, :]
                # traj_dct padding过后的dct系数
                mode_dict, traj_dct, traj_dct_cond, vel_acc_pad = sample_preprocessing(traj_tmp, cfg, mode='metrics')
                if cfg.generator == 'diffusion':
                    sampled_motion = generator_model.sample_ddim(model_select,
                                                                 traj_dct,
                                                                 traj_dct_cond,
                                                                 mode_dict,
                                                                 vel_acc_pad=vel_acc_pad)
                elif cfg.generator == 'flow_matching':
                    generator_model.update_sampling_model(model)
                    sampled_motion = generator_model.sample_fm(mode_dict, traj_dct, traj_dct_cond)
                if cfg.use_dct:
                    traj_est_tmp = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
                else:
                    traj_est_tmp = sampled_motion
                if s != slice_num - 1:
                    traj_est[s * n:(s + 1) * n, :, :] = traj_est_tmp
                else:
                    traj_est[s * n:, :, :] = traj_est_tmp

        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est


    if multimodal_dict is None:
        gt_group = None
        data_group = None
        traj_gt_arr = None
        num_samples = None
    else:
        gt_group = multimodal_dict['gt_group']
        data_group = multimodal_dict['data_group']
        traj_gt_arr = multimodal_dict['traj_gt_arr']
        num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    stats_meter = {x: {y: AverageMeter() for y in ['HumanMAC']} for x in stats_names}

    K = 50
    pred = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        if cfg.use_dct:
            mode_ = 'whole'
        else:
            mode_ = 'sliced'
        pred_i_nd = get_prediction(data_group, model, mode=mode_, slice_num=5, cfg=cfg)
        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            # pred [50, 5187, 125, 48] in h36m
            pred = pred[:, :, cfg.t_his:, :]
            # Use GPU to accelerate
            try:
                gt_group = torch.from_numpy(gt_group).to('cuda')
            except:
                pass
            try:
                pred = torch.from_numpy(pred).to('cuda')
            except:
                pass
            # pred [50, 5187, 100, 48]
            for j in range(0, num_samples):
                apd, ade, fde, mmade, mmfde = compute_all_metrics(pred[:, j, :, :],
                                                                  gt_group[j][np.newaxis, ...],
                                                                  traj_gt_arr[j])
                stats_meter['APD']['HumanMAC'].update(apd)
                stats_meter['ADE']['HumanMAC'].update(ade)
                stats_meter['FDE']['HumanMAC'].update(fde)
                stats_meter['MMADE']['HumanMAC'].update(mmade)
                stats_meter['MMFDE']['HumanMAC'].update(mmfde)
            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join(
                    [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
                )
                logger.info(str_stats)
            if save_results:
                np_file = '%s/prediction_results.npy'
                pred = pred.cpu().numpy()
                np.save(np_file % cfg.result_dir, pred)
            pred = []

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['HumanMAC'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['HumanMAC'] = new_meter['HumanMAC'].cpu().numpy()
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1['HumanMAC']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)
