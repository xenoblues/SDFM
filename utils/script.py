import copy
import os
import time

import numpy as np
import torch

from models.flow_mathcing import FlowMatching
from models.networks import MotionTransformer
from utils import padding_traj
from utils.visualization import render_animation

from data_loader.dataset_h36m import DatasetH36M
from data_loader.dataset_humaneva import DatasetHumanEva
from data_loader.dataset_h36m_multimodal import DatasetH36M_multi
from data_loader.dataset_humaneva_multimodal import DatasetHumanEva_multi
from data_loader.dataset_assemble import DatasetAsb
from scipy.spatial.distance import pdist, squareform
import multiprocessing

def create_model_and_fm(cfg):
    """
    create TransLinear model and Diffusion
    """
    if cfg.use_dct:
        num_frames = cfg.n_pre
    else:
        if not cfg.autoregression:
            num_frames = cfg.t_total
        else:
            num_frames = cfg.clip_total

    if cfg.model_name == 'MotionTransformer':
            model = MotionTransformer(
                input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                cross_attention=cfg.cross_attention,
                stylization_block=cfg.stylization_block,
                flash_attention=cfg.flash_attention,
                se_layer=cfg.se_layer,
                skip_type=cfg.skip_type
            ).to(cfg.device)

    if cfg.generator == 'flow_matching':
        generator = FlowMatching(model, cfg)

    return model, generator


def dataset_split(cfg):
    """
    output: dataset_dict, dataset_multi_test
    dataset_dict has two keys: 'train', 'test' for enumeration in train and validation.
    dataset_multi_test is used to create multi-modal data for metrics.
    """
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    if cfg.dataset == 'assemble':
        dataset_cls = DatasetAsb
        dataset = dataset_cls('train', cfg.t_his, cfg.t_pred)
        dataset_test = dataset_cls('test', cfg.t_his, cfg.t_pred)
        return {'train': dataset, 'test': dataset_test}, None

    dataset = dataset_cls('train', cfg.t_his, cfg.t_pred, actions='all')
    dataset_test = dataset_cls('test', cfg.t_his, cfg.t_pred, actions='all')
    dataset_cls_multi = DatasetH36M_multi if cfg.dataset == 'h36m' else DatasetHumanEva_multi
    dataset_multi_test = dataset_cls_multi('test', cfg.t_his, cfg.t_pred,
                                           multimodal_path=cfg.multimodal_path,
                                           data_candi_path=cfg.data_candi_path)

    return {'train': dataset, 'test': dataset_test}, dataset_multi_test


def get_multimodal_gt_full(logger, dataset_multi_test, args, cfg):
    """
    calculate the multi-modal data
    """
    logger.info('preparing full evaluation dataset...')
    data_group = []
    num_samples = 0
    data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)
    for data, _ in data_gen_multi_test:
        num_samples += 1
        data_group.append(data)
    data_group = np.concatenate(data_group, axis=0)
    all_data = data_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)
    gt_group = all_data[:, cfg.t_his:, :]

    all_start_pose = all_data[:, cfg.t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []

    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, cfg.t_his:, :])
        num_mult.append(len(ind[0]))

    # np.savez_compressed('./data/data_3d_h36m_test.npz',data=all_data)
    # np.savez_compressed('./data/data_3d_humaneva15_test.npz',data=all_data)
    num_mult = np.array(num_mult)
    logger.info('=' * 80)
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    logger.info('done...')
    logger.info('=' * 80)
    return {'traj_gt_arr': traj_gt_arr,
            'data_group': data_group,
            'gt_group': gt_group,
            'num_samples': num_samples}


def get_assemble_test(logger, dataset_test, args, cfg):
    """
    calculate the multi-modal data
    """
    logger.info('preparing full evaluation dataset...')
    data_group = []
    num_samples = 0
    data_gen_test = dataset_test.iter_generator(step=cfg.t_his)
    for data in data_gen_test:
        num_samples += 1
        data_group.append(data)
    data_group = np.asarray(data_group)
    all_data = data_group.reshape(data_group.shape[0], data_group.shape[1], -1)
    gt_group = all_data[:, cfg.t_his:, :]

    all_start_pose = all_data[:, cfg.t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []

    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, cfg.t_his:, :])
        num_mult.append(len(ind[0]))

    # np.savez_compressed('./data/data_3d_h36m_test.npz',data=all_data)
    # np.savez_compressed('./data/data_3d_humaneva15_test.npz',data=all_data)

    num_mult = np.array(num_mult)
    logger.info('=' * 80)
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    logger.info('done...')
    logger.info('=' * 80)
    return {'traj_gt_arr': traj_gt_arr,
            'data_group': data_group,
            'gt_group': gt_group,
            'num_samples': num_samples}


def display_exp_setting(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 80)
    log_dict = cfg.__dict__.copy()
    for key in list(log_dict):
        if 'dir' in key or 'path' in key or 'dct' in key:
            del log_dict[key]
    del log_dict['zero_index']
    del log_dict['idx_pad']
    logger.info(log_dict)
    logger.info('=' * 80)


def sample_preprocessing(traj, cfg, mode):
    """
    This function is used to preprocess traj for sample_ddim().
    input : traj_seq, cfg, mode
    output: a dict for specific mode,
            traj_dct,
            traj_dct_mod
    """
    vel_acc_pad = None
    if mode.split('_')[0] == 'fix':

        # skeleton joints in h36m
        fix_list = [
            [0, 1, 2],  #
            [3, 4, 5],
            [6, 7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        ]

        index = int(mode.split('_')[1])

        # [ ['right_leg'], ['left_leg'], ['torso'], [‘left_arm’], ['right_arm'] ]
        joint_fix_lb = fix_list[index][0] * 3
        joint_fix_ub = fix_list[index][-1] * 3 + 3

        traj_fix = traj[:, cfg.idx_pad, :]
        traj_fix[:, :, joint_fix_lb:joint_fix_ub] = traj[:, :, joint_fix_lb:joint_fix_ub]

        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        mask_fix = copy.deepcopy(mask)
        mask_fix[:, :, joint_fix_lb:joint_fix_ub] = 1

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)

        if cfg.use_dct:
            traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)


        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'traj_fix': traj_fix,
                'mask': mask_fix,
                'sample_num': n,
                'mode': 'control'}, traj_dct, traj_dct_mod, vel_acc_pad

    elif mode == 'switch':
        n = traj.shape[0]
        traj_switch = traj[:, (cfg.t_pred - cfg.t_his):(cfg.t_pred + cfg.t_his), :]
        direct_current = traj[:, (cfg.t_pred - cfg.t_his), :].unsqueeze(1).repeat(1, cfg.t_pred - cfg.t_his,
                                                                                  1)
        traj_switch = torch.cat([direct_current, traj_switch], dim=1)

        traj_switch = traj_switch[0].unsqueeze(0).repeat(n, 1, 1)
        # traj_switch = torch.roll(traj_switch, 1, 0) # make the target traj be various

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        mask_end = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(cfg.t_pred - cfg.t_his, cfg.t_his + cfg.t_pred):
            mask_end[:, i, :] = 1

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)

        if cfg.use_dct:
            traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'traj_switch': traj_switch,
                'mask_end': mask_end,
                'mask': mask,
                'sample_num': n,
                'mode': 'switch'}, traj_dct, traj_dct_mod, vel_acc_pad

    elif mode == 'gif':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)

        if cfg.use_dct:
            traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'gif'}, traj_dct, traj_dct_mod, vel_acc_pad

    elif mode == 'pred':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)

        if cfg.use_dct:
            traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'pred'}, traj_dct, traj_dct_mod, vel_acc_pad

    elif mode == 'zero_shot':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)

        if cfg.use_dct:
            traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'zero_shot'}, traj_dct, traj_dct_mod, vel_acc_pad

    elif mode == 'metrics':
        n = traj.shape[0]

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):  # 历史帧标记为1
            mask[:, i, :] = 1

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)  # last frame padding

        if cfg.use_dct:
            traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)  # 压缩到n_pre帧
            traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'metrics'}, traj_dct, traj_dct_mod, vel_acc_pad

    else:
        raise NotImplementedError(f"unknown purpose for sampling: {mode}")
