import copy
import time

import numpy as np
import torch
from flow_matching.solver import ODESolver
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel

from utils.visualization import render_animation
from models.networks import EMA
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch.nn.modules import Module
from flow_matching.utils import ModelWrapper
from adan import Adan

def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    t = 1 / (1 + sigma)
    t = torch.clip(t, min=0.0001, max=1.0)
    return t


class Trainer_fm:
    def __init__(self,
                 model,
                 generator,
                 dataset,
                 cfg,
                 multimodal_dict,
                 logger,
                 tb_logger):
        super().__init__()

        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None
        self.val_min_loss = None

        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None
        self.his_mask = None


        self.model = model
        self.generator = generator
        self.dataset = dataset
        self.multimodal_dict = multimodal_dict
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger
        self.iter = 0
        self.lrs = []

        self.resume = False

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False).cuda()
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

        # flow matching
        self.path = CondOTProbPath()


    def loop(self):
        self.before_train()
        if self.iter == -1:
            self.iter = 0
        for self.iter in range(self.iter, self.cfg.num_epoch):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()

    def before_train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        # self.optimizer = Adan(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.02)

        if self.cfg.frame_mask:
            self.criterion = nn.MSELoss(reduction='sum')
        else:
            self.criterion = nn.MSELoss()

        self.iter = -1
        if self.cfg.resume:
            loaded_ckpt = torch.load(self.cfg.ckpt_path, map_location='cuda')
            self.model.load_state_dict(loaded_ckpt, strict=False)
            if self.cfg.ckpt_path[-6] == '_':
                self.iter = int(self.cfg.ckpt_path[-5:-3])
            else:
                self.iter = int(self.cfg.ckpt_path[-6:-3])
            milestone = np.asarray(self.cfg.milestone)
            if self.iter > milestone[-1]:
                power = milestone.shape[0]
            else:
                power = int(min(np.argwhere(milestone > self.iter)))
            last_lr = self.cfg.lr * (self.cfg.gamma ** power)
            self.optimizer = optim.Adam([{"params": self.model.parameters(), "initial_lr": self.cfg.lr}],
                                        lr=last_lr)

        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,
                                                           gamma=self.cfg.gamma, last_epoch=self.iter)
        self.val_min_loss = MinMeter()
        self.his_mask = torch.zeros([self.cfg.batch_size, self.cfg.t_his + self.cfg.t_pred, self.dataset['train'].traj_dim]).to(self.cfg.device)
        for i in range(0, self.cfg.t_his):
            self.his_mask[:, i, :] = 1

    def before_train_step(self):
        self.model.train()
        self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
                                                                        batch_size=self.cfg.batch_size)
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):
        train_n_pre = self.cfg.n_pre + 5
        for traj_np, mask in self.generator_train:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate


                traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)

                vel_acc_pad = None
                if self.cfg.residual_data:  # 使用速度作为输入
                    # res_traj = torch.zeros(traj.shape, device=self.cfg.device, dtype=self.cfg.dtype)
                    # res_traj[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]
                    # traj = res_traj

                    # B, T-1, V, 3
                    if np.random.random() > self.cfg.mod_train:
                        vel_acc = None
                    else:
                        vel_acc = cal_vel_acc(traj)
                        vel_acc_pad = padding_vel(vel_acc, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                traj_dct = traj.clone()
                traj_mod = traj_pad
                input_traj = None

                if self.cfg.random_sample:
                    input_traj = torch.zeros((traj.shape[0], self.cfg.n_pre, traj.shape[-1]))
                    for i in range(traj.shape[0]):
                        a = np.arange(traj.shape[1])
                        np.random.shuffle(a)
                        a = np.sort(a[:self.cfg.n_pre])
                        input_traj[i, :, :] = traj[i, a, :]

                if self.cfg.use_dct:
                    traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                    traj_pad_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                    if np.random.random() > self.cfg.mod_train:
                        traj_mod = None
                    else:
                        traj_mod = traj_pad_dct
                    noise = torch.randn(traj_dct.shape).to(self.cfg.device)
                    traj_pad_dct_noised = traj_pad_dct + noise
                    # traj_pad_noised = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], traj_pad_dct_noised[:, :self.cfg.n_pre])
                    # input_traj = torch.mul(self.his_mask, traj) + torch.mul(1 - self.his_mask, traj_pad_noised)
                    # input_traj = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], input_traj)
                    input_traj = noise
                else:
                    noise = torch.randn(traj_pad.shape).to(self.cfg.device)
                    input_traj = noise
                    if np.random.random() > self.cfg.mod_train:
                        traj_mod = None

            # train

            if self.cfg.skewed_timesteps:
                t = skewed_timestep_sample(input_traj.shape[0], device=self.cfg.device)
            else:
                t = torch.rand(input_traj.shape[0]).to(self.cfg.device)
            path_sample = self.path.sample(t=t, x_0=input_traj, x_1=traj_dct)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            loss = torch.pow(self.model(x_t, t, traj_mod) - u_t, 2).mean()

            self.optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():  # 反向传播时：在求导时开启侦测
            loss.backward()
            self.optimizer.step()


            # if self.iter >= 5:
            #     for name, params in self.model.named_parameters():
            #         if params.requires_grad:
            #             print("name", name, 'params:', params, "grad:", params.grad)

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]

            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)
            end_time5 = time.time()
            # print("epoch耗时:{:.5f}秒".format(end_time5 - start_time))

            del loss, traj, traj_dct, traj_mod, traj_pad, traj_np, mask

    def after_train_step(self):
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg,
                                                                            self.lrs[-1]))

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
                                                                     batch_size=self.cfg.batch_size * 4)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for traj_np, mask in self.generator_val:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                vel_acc_pad = None
                if self.cfg.residual_data:  # 使用速度作为输入
                    # res_traj = torch.zeros(traj.shape, device=self.cfg.device, dtype=self.cfg.dtype)
                    # res_traj[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]
                    # traj = res_traj
                    # B, T-1, V, 3
                    if np.random.random() > self.cfg.mod_train:
                        vel_acc_pad = None
                    else:
                        vel_acc = cal_vel_acc(traj)
                        vel_acc_pad = padding_vel(vel_acc, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)  #
                # [n_pre × (t_his + t_pre)] matmul [(t_his + t_pre) × 3 * (joints - 1)]
                traj_dct = traj
                traj_dct_mod = traj_pad

                if self.cfg.use_dct:
                    traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                    traj_pad_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                    if np.random.random() > self.cfg.mod_train :
                        traj_dct_mod = None
                    else:
                        traj_dct_mod = traj_pad_dct
                    noise = torch.randn(traj_dct.shape).to(self.cfg.device)
                    traj_pad_dct_noised = traj_pad_dct + noise
                    # traj_pad_noised = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], traj_pad_dct_noised[:, :self.cfg.n_pre])
                    # input_traj = torch.mul(self.his_mask, traj) + torch.mul(1 - self.his_mask, traj_pad_noised)
                    # input_traj = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], input_traj)
                    input_traj = noise
                else:
                    noise = torch.randn(traj_dct.shape).to(self.cfg.device)
                    input_traj = noise
                    if np.random.random() > self.cfg.mod_train:
                        traj_dct_mod = None

                if self.cfg.skewed_timesteps:
                    t = skewed_timestep_sample(input_traj.shape[0], device=self.cfg.device)
                else:
                    t = torch.rand(input_traj.shape[0]).to(self.cfg.device)
                path_sample = self.path.sample(t=t, x_0=input_traj, x_1=traj_dct)
                x_t = path_sample.x_t
                u_t = path_sample.dx_t

                loss = torch.pow(self.model(x_t, t, traj_dct_mod) - u_t, 2).mean()

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np, input_traj

    def after_val_step(self):
        self.val_min_loss.update(self.iter, self.val_losses.avg)
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(self.iter, time.time() - self.t_s, self.val_losses.avg))
        self.logger.info('====> Min Val Loss: {} Epoch: {}'.format(self.val_min_loss.min_loss, self.val_min_loss.min_iter))
        if self.iter % self.cfg.save_gif_interval == 0:
            if self.cfg.ema is True:
                pose_gen = pose_generator(self.dataset['test'], self.ema_model, self.generator, self.cfg, mode='gif')
            else:
                pose_gen = pose_generator(self.dataset['test'], self.model, self.generator, self.cfg, mode='gif')
            render_animation(self.dataset['test'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'))

        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))

        if (self.iter + 1) >= 400 and self.iter == self.val_min_loss.min_iter:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))
            self.logger.info('====> Save Current Min Val Loss Epoch: {}'.format(self.val_min_loss.min_iter))

        if self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0:
            if self.cfg.ema is True:
                compute_stats(self.generator, self.multimodal_dict, self.ema_model, self.logger, self.cfg)
            else:
                compute_stats(self.generator, self.multimodal_dict, self.model, self.logger, self.cfg)
