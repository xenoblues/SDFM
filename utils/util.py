import os
import random

import torch as tr
import numpy as np


def seed_set(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)


def generate_pad(padding, t_his, t_pred):
    zero_index = None
    if padding == 'Zero':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        zero_index = max(idx_pad)
    elif padding == 'Repeat':
        idx_pad = list(range(t_his)) * int(((t_pred + t_his) / t_his))
        # [0, 1, 2,....,24, 0, 1, 2,....,24, 0, 1, 2,...., 24...]
    elif padding == 'LastFrame':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        # [0, 1, 2,....,24, 24, 24,.....]
    else:
        raise NotImplementedError(f"unknown padding method: {padding}")
    return idx_pad, zero_index


def padding_traj(traj, padding, idx_pad, zero_index):
    if padding == 'Zero':
        traj_tmp = traj
        traj_tmp[..., zero_index, :] = 0
        traj_pad = traj_tmp[..., idx_pad, :]
    else:
        traj_pad = traj[..., idx_pad, :]

    return traj_pad


def padding_vel(vel, padding, idx_pad, zero_index):
    if padding == 'Zero':
        vel_tmp = vel
        vel_tmp[..., zero_index, :, :] = 0
        vel_pad = vel_tmp[..., idx_pad, :, :]
    else:
        vel_pad = vel[..., idx_pad, :, :]

    return vel_pad


def post_process(pred, cfg):
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
    pred = np.concatenate((np.tile(np.zeros((1, cfg.t_his + cfg.t_pred, 1, 3)), (pred.shape[0], 1, 1, 1)), pred),
                          axis=2)
    pred[..., :1, :] = 0
    return pred


def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = tr.from_numpy(dct_m)
        idct_m = tr.from_numpy(idct_m)
    return dct_m, idct_m


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = tr.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tr.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * tr.sqrt(distances)

    return distances


def _pairwise_distances_l1(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    distances = tr.abs(embeddings[None, :, :] - embeddings[:, None, :])
    return distances


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R


def absolute2relative(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / np.linalg.norm(xt, axis=-1, keepdims=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = np.linalg.norm(x0[..., 1:, :] - x0[..., parents[1:], :], axis=-1, keepdims=True)
        xt = x * limb_l
        xt0 = np.zeros_like(xt[..., :1, :])
        xt = np.concatenate([xt0, xt], axis=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt


def absolute2relative_torch(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / tr.norm(xt, dim=-1, keepdim=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = tr.norm(x0[..., 1:, :] - x0[..., parents[1:], :], dim=-1, keepdim=True)
        xt = x * limb_l
        xt0 = tr.zeros_like(xt[..., :1, :])
        xt = tr.cat([xt0, xt], dim=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def multiscale_filters(A, num):
    # L 图的拉普拉斯矩阵
    I = np.identity(A.shape[0])
    # for i in range(L.shape):
    # I[i, i] = 1
    # A_hat = 1/2 * (I + L)
    # T = I - A_hat
    filters = [A]
    assert num > 0
    for i in range(1, num):
        filters.append(filters[i - 1] ** (2 ** (i - 1)) - filters[i - 1] ** (2 ** i))
    return np.asarray(filters)


def get_temporal_graph(num_node):
    A = np.eye(num_node, dtype=float)
    for i in range(num_node):
        if i - 1 >= 0:
            A[i, i - 1] = 1
        if i + 1 < num_node:
            A[i, i + 1] = 1
    A = normalize_digraph(A)
    return A


def cal_vel_acc(traj):
    traj_tmp = traj.clone().reshape([traj.shape[0], traj.shape[1], -1, 3])[:, :-1, :, :]
    traj_tmp2 = traj.clone().reshape([traj.shape[0], traj.shape[1], -1, 3])[:, 1:, :, :]
    vel = tr.linalg.norm(traj_tmp2 - traj_tmp, dim=-1).unsqueeze(-1)
    acc = vel[:, 1:, :, :] - vel[:, :-1, :, :]
    vel = tr.cat((vel, vel[:, -1:, :, :]), dim=1)
    acc = tr.cat((acc, acc[:, -1:, :, :], acc[:, -1:, :, :]), dim=1)
    vel_acc = tr.cat((vel, acc), dim=-1)
    return vel_acc


def get_time_discretization(nfes: int, rho=7):
    step_indices = tr.arange(nfes, dtype=tr.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = tr.cat([sigma_vec, tr.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - tr.clip(time_vec, min=0.0, max=1.0)
    return t_samples

