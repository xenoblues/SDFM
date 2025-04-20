import copy
import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from utils import *
import numpy as np
import math
from copy import deepcopy


class SamplingModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float=None, dct_mod: torch.Tensor=None):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )

        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale == 0.0 or cfg_scale is None:
            with torch.no_grad():
                result = self.model(x, t)
        elif cfg_scale == 1.0:
            with torch.no_grad():
                result = self.model(x, t, dct_mod)
        else:
            with torch.no_grad():
                B = x.shape[0]
                conditional = self.model(x, t, dct_mod)
                condition_free = self.model(x, t)
                result = cfg_scale * conditional + (1.0 - cfg_scale) * condition_free

                # parallel compute conditional and unconditional generation
                # x_ = x.clone()
                # x_double = torch.concatenate((x, x_), dim=0)
                # t_ = t.clone()
                # t_double = torch.concatenate((t, t_), dim=0)
                # dct_mod_zero = torch.zeros_like(dct_mod)
                # dct_mod_double = torch.concatenate((dct_mod, dct_mod_zero), dim=0)
                # y = self.model(x_double, t_double, dct_mod_double)
                # result = cfg_scale * y[:B, :, :] + (1 - cfg_scale) * y[B:, :, :]

        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


class FlowMatching:
    def __init__(self, net_model, cfg):
        self.sampling_model = SamplingModel(net_model)
        self.cfg = cfg

    def update_sampling_model(self, new_model):
        self.sampling_model.model = new_model

    @staticmethod
    def grid_constructor(t, step_size):
        start_time = t[0]
        end_time = t[-1]

        niters = torch.ceil((end_time - start_time) / step_size + 1).item()
        t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
        t_infer[-1] = t[-1]
        return t_infer

    def traj_fusion(self, traj_ori, y_t, z_dct, t, mask, b_dct=True):  # 拼接原始历史
        z_temp = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], z_dct[:, :self.cfg.n_pre])
        if b_dct:
            y_t_ori = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], y_t[:, :self.cfg.n_pre])  # iDCT变换至时域
        else:
            y_t_ori = y_t
        y_mid = torch.mul(mask, traj_ori) + torch.mul((1 - mask), y_t_ori)  # mask拼接原始历史
        if b_dct:
            y_mid = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], y_mid)  # DCT变换到n_pre帧
        return y_mid

    def sample_fm(self, mode_dict, traj_dct, traj_dct_mod, fusion_traj_till=1.0, mod_till=1.0):
        with torch.set_grad_enabled(False):
            if self.cfg.edm_schedule:
                t = get_time_discretization(nfes=self.cfg.ode_options["nfe"])
            else:
                t = torch.tensor([0.0, 1.0], device=self.cfg.device)

            step_size = self.cfg.ode_options["step_size"] if "step_size" in self.cfg.ode_options else None
            assert step_size is not None

            time_grid = self.grid_constructor(t, step_size)
            assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

            z_dct = torch.randn_like(traj_dct).to(traj_dct.device)  # dct field noise
            y0 = z_dct
            # y0 = traj_dct + z_dct
            # y0 = self.traj_fusion(traj_pad, y0, z, mode_dict['mask'])
            solution = torch.empty(len(time_grid), * y0.shape, dtype=y0.dtype, device=y0.device)
            solution[0] = y0
            traj_pad = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], traj_dct_mod[:, :self.cfg.n_pre])   # iDCT变换至时域

            j = 1
            for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
                dt = t1 - t0
                if self.cfg.ode_method == 'euler':
                    if t0 <= mod_till:
                        dy = dt * self.sampling_model(y0, t0, cfg_scale=self.cfg.cfg_scale, dct_mod=traj_dct_mod)
                    else:
                        dy = dt * self.sampling_model(y0, t0)
                    y1 = y0 + dy
                    if t0 <= fusion_traj_till:
                        y1 = self.traj_fusion(traj_pad, y1, z_dct, t0, mode_dict['mask'], self.cfg.use_dct)
                    y0 = y1
                elif self.cfg.ode_method == 'midpoint':
                    half_dt = 0.5 * dt
                    f0 = self.sampling_model(y0, t0, cfg_scale=self.cfg.cfg_scale, dct_mod=traj_dct_mod)
                    y_mid = y0 + f0 * half_dt
                    if t0 <= fusion_traj_till:
                        y_mid = self.traj_fusion(traj_pad, y_mid, z_dct, t0, mode_dict['mask'])
                    dy = dt * self.sampling_model(y_mid, t0 + half_dt, dct_mod=traj_dct_mod)
                    y1 = y0 + dy
                    if t0 <= fusion_traj_till:
                        y1 = self.traj_fusion(traj_pad, y1, z_dct, t0, mode_dict['mask'])
                    y0 = y1
            sampled_motion = y0
            return sampled_motion
